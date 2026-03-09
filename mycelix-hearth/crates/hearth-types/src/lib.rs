//! Shared types for the Mycelix Hearth (Family/Household) cluster.
//!
//! These types are used across all Hearth zomes for consistent
//! data modeling and cross-zome communication.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Hearth Types
// ============================================================================

/// Type of hearth (family/household structure).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HearthType {
    Nuclear,
    Extended,
    Chosen,
    Blended,
    Multigenerational,
    Intentional,
    CoPod,
    Custom(String),
}

/// Role of a member within a hearth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    /// Original creator of the hearth.
    Founder,
    /// Respected elder with advisory authority.
    Elder,
    /// Full adult member with voting rights.
    Adult,
    /// Older minor with some autonomy.
    Youth,
    /// Young child, fully dependent.
    Child,
    /// Temporary or limited-access member.
    Guest,
    /// Departed member preserved in memory.
    Ancestor,
}

impl MemberRole {
    /// Whether this role has guardian-level authority.
    pub fn is_guardian(&self) -> bool {
        matches!(
            self,
            MemberRole::Founder | MemberRole::Elder | MemberRole::Adult
        )
    }

    /// Whether this role represents a minor.
    pub fn is_minor(&self) -> bool {
        matches!(self, MemberRole::Youth | MemberRole::Child)
    }

    /// Default vote weight in basis points for this role.
    pub fn default_vote_weight_bp(&self) -> u32 {
        match self {
            MemberRole::Founder => 10000,
            MemberRole::Elder => 10000,
            MemberRole::Adult => 10000,
            MemberRole::Youth => 5000,
            MemberRole::Child => 0,
            MemberRole::Guest => 0,
            MemberRole::Ancestor => 0,
        }
    }
}

/// Status of a hearth membership.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MembershipStatus {
    Active,
    Invited,
    Departed,
    Ancestral,
}

/// Type of kinship bond between two members.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BondType {
    Parent,
    Child,
    Sibling,
    Partner,
    Grandparent,
    Grandchild,
    AuntUncle,
    NieceNephew,
    Cousin,
    ChosenFamily,
    Guardian,
    Ward,
    Custom(String),
}

// ============================================================================
// Autonomy Types
// ============================================================================

/// Graduated autonomy tier (not binary — supports progressive independence).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyTier {
    /// Full dependency on guardians.
    Dependent,
    /// Can act with direct supervision.
    Supervised,
    /// Can act with guidance available.
    Guided,
    /// Can act independently in most areas.
    SemiAutonomous,
    /// Full autonomy — adult-equivalent.
    Autonomous,
}

// ============================================================================
// Visibility & Privacy
// ============================================================================

/// Privacy scope for hearth entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HearthVisibility {
    /// Visible to all hearth members.
    AllMembers,
    /// Visible only to Adult/Elder/Founder roles.
    AdultsOnly,
    /// Visible only to guardians of a specific member.
    GuardiansOnly,
    /// Visible only to specified agents.
    Specified(Vec<AgentPubKey>),
}

// ============================================================================
// Care Types
// ============================================================================

/// Category of care activity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CareType {
    Childcare,
    Eldercare,
    PetCare,
    Chore,
    MealPrep,
    Medical,
    Emotional,
    Custom(String),
}

/// Recurrence schedule for care tasks and rhythms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Recurrence {
    Daily,
    Weekly,
    Monthly,
    Custom(String),
}

// ============================================================================
// Emergency Types
// ============================================================================

/// Severity level for emergency alerts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Type of emergency alert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    Medical,
    Natural,
    Security,
    Missing,
    Fire,
    Custom(String),
}

// ============================================================================
// Gratitude Types
// ============================================================================

/// Category of gratitude expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GratitudeType {
    Appreciation,
    Acknowledgment,
    Celebration,
    Blessing,
    Custom(String),
}

// ============================================================================
// Story Types
// ============================================================================

/// Category of family story.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoryType {
    Memory,
    Tradition,
    Recipe,
    Wisdom,
    Origin,
    Migration,
    Custom(String),
}

// ============================================================================
// Milestone & Transition Types
// ============================================================================

/// Type of life milestone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MilestoneType {
    Birth,
    Birthday,
    FirstStep,
    SchoolStart,
    Graduation,
    Engagement,
    Marriage,
    NewHome,
    Retirement,
    Passing,
    Custom(String),
}

/// Type of life transition (maps to Living Primitives Liminality).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionType {
    JoiningHearth,
    LeavingHearth,
    ComingOfAge,
    Retirement,
    Bereavement,
    Custom(String),
}

/// Phase within a liminal transition (forward-only progression).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TransitionPhase {
    PreLiminal,
    Liminal,
    PostLiminal,
    Integrated,
}

// ============================================================================
// Decision Types
// ============================================================================

/// Method for reaching a family decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionType {
    Consensus,
    MajorityVote,
    ElderDecision,
    GuardianDecision,
}

/// Status of a decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionStatus {
    Open,
    Closed,
    Finalized,
}

// ============================================================================
// Resource Types
// ============================================================================

/// Category of shared resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Tool,
    Vehicle,
    Book,
    Kitchen,
    Electronics,
    Clothing,
    Custom(String),
}

/// Status of a resource loan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoanStatus {
    Active,
    Returned,
    Overdue,
}

// ============================================================================
// Rhythm Types
// ============================================================================

/// Type of family rhythm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhythmType {
    Morning,
    Evening,
    Weekly,
    Seasonal,
    Custom(String),
}

/// Presence status of a hearth member.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceStatusType {
    Home,
    Away,
    Working,
    Sleeping,
    DoNotDisturb,
}

// ============================================================================
// Invitation Status
// ============================================================================

/// Status of a hearth invitation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvitationStatus {
    Pending,
    Accepted,
    Declined,
    Expired,
}

// ============================================================================
// Safety Check-In
// ============================================================================

/// Status reported during emergency check-in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyStatus {
    Safe,
    NeedHelp,
    NoResponse,
}

// ============================================================================
// Care Swap Status
// ============================================================================

/// Status of a care task swap request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapStatus {
    Proposed,
    Accepted,
    Declined,
    Completed,
}

// ============================================================================
// Care Schedule Status
// ============================================================================

/// Status of a care schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CareScheduleStatus {
    Active,
    Paused,
    Completed,
}

// ============================================================================
// Appreciation Circle Status
// ============================================================================

/// Status of an appreciation circle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircleStatus {
    Open,
    InProgress,
    Completed,
}

// ============================================================================
// Autonomy Request Status
// ============================================================================

/// Status of an autonomy capability request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyRequestStatus {
    Pending,
    Approved,
    Denied,
}

// ============================================================================
// Bond Strength — Fixed-Point Math (H1: No f64 in consensus paths)
// ============================================================================

/// Basis-point bond strength (0 = dissolved, 10000 = maximum).
/// Using u32 instead of f64 ensures deterministic results across
/// all WASM runtimes (ARM, x86) — no floating-point divergence.
pub type BondStrength = u32;

/// Maximum bond strength (1.0 in basis points).
pub const BOND_MAX: u32 = 10000;

/// Default starting strength for family bonds (0.7).
pub const BOND_BASE_FAMILY: u32 = 7000;

/// Minimum bond strength — bonds don't fully dissolve (0.1).
pub const BOND_MIN: u32 = 1000;

/// Pre-computed decay table for deterministic cross-node consensus.
/// Maps days_inactive to strength_remaining (basis points).
/// Pre-calculated from e^(-0.02 * days) * 10000, rounded to nearest integer.
pub const DECAY_TABLE: &[(u32, u32)] = &[
    (0, 10000),
    (1, 9802),
    (7, 8694),
    (14, 7558),
    (30, 5488),
    (60, 3012),
    (90, 1653),
    (120, 907),
    (180, 273),
    (270, 45),
    (365, 7),
];

/// Deterministic decay: linear interpolation between table entries.
/// Uses integer-only math for consensus safety.
pub fn decayed_strength(initial_bp: u32, days_inactive: u32) -> u32 {
    if days_inactive == 0 {
        return initial_bp;
    }

    // Bonds already at or below BOND_MIN don't decay further.
    // This prevents the BOND_MIN floor from inflating sub-minimum bonds.
    if initial_bp < BOND_MIN {
        return initial_bp;
    }

    // Find the bracketing table entries
    let mut lower_days = 0u32;
    let mut lower_bp = BOND_MAX;
    let mut upper_days = 365u32;
    let mut upper_bp = 7u32;

    for &(d, bp) in DECAY_TABLE {
        if d <= days_inactive {
            lower_days = d;
            lower_bp = bp;
        }
        if d >= days_inactive {
            upper_days = d;
            upper_bp = bp;
            break;
        }
    }

    // Beyond table: clamp to minimum
    if days_inactive >= 365 {
        let factor = DECAY_TABLE.last().map(|&(_, bp)| bp).unwrap_or(7);
        let result = (initial_bp as u64 * factor as u64 / BOND_MAX as u64) as u32;
        return result.max(BOND_MIN).min(initial_bp);
    }

    // Exact table hit
    if lower_days == upper_days || lower_days == days_inactive {
        let result = (initial_bp as u64 * lower_bp as u64 / BOND_MAX as u64) as u32;
        return result.max(BOND_MIN).min(initial_bp);
    }

    // Linear interpolation using integer math
    let day_range = upper_days - lower_days;
    let bp_range = lower_bp.saturating_sub(upper_bp);
    let day_offset = days_inactive - lower_days;

    // interpolated_factor = lower_bp - (bp_range * day_offset / day_range)
    let interpolated_factor =
        lower_bp - (bp_range as u64 * day_offset as u64 / day_range as u64) as u32;
    let result = (initial_bp as u64 * interpolated_factor as u64 / BOND_MAX as u64) as u32;
    result.max(BOND_MIN).min(initial_bp)
}

// ============================================================================
// Epoch Rollup Types (H2: Signals for daily noise, weekly DHT entries)
// ============================================================================

/// Weekly digest summarizing high-frequency activity.
/// Written to DHT once per week (or on-demand sync).
/// Uses #[hdk_entry_helper] so it can be registered as an entry type
/// in kinship integrity (the core hearth zome).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WeeklyDigest {
    pub hearth_hash: ActionHash,
    pub epoch_start: Timestamp,
    pub epoch_end: Timestamp,
    pub bond_updates: Vec<BondUpdate>,
    pub care_summary: Vec<CareSummary>,
    pub gratitude_summary: Vec<GratitudeSummary>,
    pub rhythm_summary: Vec<RhythmSummary>,
    pub created_by: AgentPubKey,
    pub created_at: Timestamp,
}

/// Bond update within a weekly digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BondUpdate {
    pub member_a: AgentPubKey,
    pub member_b: AgentPubKey,
    pub co_creation_count: u32,
    pub quality_sum_bp: u32,
}

/// Care summary within a weekly digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CareSummary {
    pub assignee: AgentPubKey,
    pub tasks_completed: u32,
    pub hours_hundredths: u32,
}

/// Gratitude summary within a weekly digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GratitudeSummary {
    pub from_agent: AgentPubKey,
    pub to_agent: AgentPubKey,
    pub count: u32,
}

/// Rhythm summary within a weekly digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RhythmSummary {
    pub rhythm_hash: ActionHash,
    pub occurrences: u32,
    pub avg_participation_bp: u32,
}

// ============================================================================
// Cross-Zome DTOs
// ============================================================================

/// Input for epoch-based digest queries used by gratitude, care, rhythms,
/// and bridge coordinators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestEpochInput {
    pub hearth_hash: ActionHash,
    pub epoch_start: Timestamp,
    pub epoch_end: Timestamp,
}

// NOTE: Anchor is NOT defined here. Each integrity zome defines its own
// Anchor entry type using #[hdk_entry_helper] (required for Holochain validation).
// Having Anchor in shared types caused ambiguity with glob imports.

// ============================================================================
// Severance Types (H3: Coming-of-age data migration)
// ============================================================================

/// Input for initiating a severance (data export on departure/coming-of-age).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeveranceInput {
    pub hearth_hash: ActionHash,
    pub member_hash: ActionHash,
    pub export_milestones: bool,
    pub export_care_history: bool,
    pub export_bond_snapshot: bool,
    pub new_role: MemberRole,
}

/// Summary of a completed severance, stored on hearth DHT as audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeveranceSummaryData {
    pub hearth_hash: ActionHash,
    pub member: AgentPubKey,
    pub milestones_exported: u32,
    pub care_records_exported: u32,
    pub bond_snapshot_exported: bool,
    pub new_role: MemberRole,
    pub completed_at: Timestamp,
}

// ============================================================================
// Signal Types (for emit_signal)
// ============================================================================

/// Signal payload for real-time events (ephemeral, not stored on DHT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HearthSignal {
    GratitudeExpressed {
        from_agent: AgentPubKey,
        to_agent: AgentPubKey,
        message: String,
        gratitude_type: GratitudeType,
    },
    CareTaskCompleted {
        assignee: AgentPubKey,
        schedule_hash: ActionHash,
        care_type: CareType,
    },
    RhythmOccurred {
        rhythm_hash: ActionHash,
        participants: Vec<AgentPubKey>,
    },
    EmergencyAlert {
        alert_hash: ActionHash,
        severity: AlertSeverity,
        message: String,
    },
    MemberJoined {
        hearth_hash: ActionHash,
        agent: AgentPubKey,
        role: MemberRole,
    },
    MemberDeparted {
        hearth_hash: ActionHash,
        agent: AgentPubKey,
    },
    BondTended {
        member_a: AgentPubKey,
        member_b: AgentPubKey,
        quality_bp: u32,
    },
    PresenceChanged {
        agent: AgentPubKey,
        status: PresenceStatusType,
    },
    /// Emitted when a best-effort cross-zome/cross-cluster call fails.
    /// Provides observability without blocking the caller.
    CrossZomeCallFailed {
        zome: String,
        function: String,
        error: String,
    },
    /// A vote was cast on a decision.
    VoteCast {
        decision_hash: ActionHash,
        voter: AgentPubKey,
        choice: u32,
    },
    /// A vote was amended (old vote replaced with new).
    VoteAmended {
        decision_hash: ActionHash,
        voter: AgentPubKey,
        old_choice: u32,
        new_choice: u32,
    },
    /// A decision was manually closed before its deadline.
    DecisionClosed {
        decision_hash: ActionHash,
        closed_by: AgentPubKey,
    },
    /// A decision was finalized with a winning option.
    DecisionFinalized {
        decision_hash: ActionHash,
        chosen_option: u32,
        participation_rate_bp: u32,
    },
    /// A new family story was created.
    StoryCreated {
        hearth_hash: ActionHash,
        story_hash: ActionHash,
        storyteller: AgentPubKey,
        story_type: StoryType,
    },
    /// A story was updated by its storyteller.
    StoryUpdated {
        story_hash: ActionHash,
        updated_by: AgentPubKey,
    },
    /// A family tradition was observed.
    TraditionObserved {
        tradition_hash: ActionHash,
        observed_by: AgentPubKey,
    },
    /// A shared resource was lent to a member.
    ResourceLent {
        resource_hash: ActionHash,
        borrower: AgentPubKey,
        due_date: Timestamp,
    },
    /// A borrowed resource was returned.
    ResourceReturned {
        loan_hash: ActionHash,
        borrower: AgentPubKey,
    },
    /// An expense was logged against a budget category.
    ExpenseLogged {
        budget_hash: ActionHash,
        amount_cents: u64,
    },
    /// A milestone was recorded in a family timeline.
    MilestoneRecorded {
        hearth_hash: ActionHash,
        milestone_hash: ActionHash,
        milestone_type: MilestoneType,
    },
    /// A life transition advanced to a new phase.
    TransitionAdvanced {
        transition_hash: ActionHash,
        new_phase: TransitionPhase,
    },
    /// An autonomy tier was advanced (guardian action).
    TierAdvanced {
        profile_hash: ActionHash,
        from_tier: AutonomyTier,
        to_tier: AutonomyTier,
    },
    /// A capability request was approved by a guardian.
    CapabilityApproved {
        request_hash: ActionHash,
        capability: String,
    },
    /// A capability request was denied by a guardian.
    CapabilityDenied {
        request_hash: ActionHash,
        capability: String,
    },
    /// A tier transition was progressed to the next phase.
    TransitionProgressed {
        transition_hash: ActionHash,
        new_phase: TransitionPhase,
    },
    /// A care swap was accepted by the responder.
    SwapAccepted {
        swap_hash: ActionHash,
        hearth_hash: ActionHash,
    },
    /// A care swap was declined by the responder.
    SwapDeclined {
        swap_hash: ActionHash,
        hearth_hash: ActionHash,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper constructors for fake Holochain types
    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xAAu8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xBBu8; 36])
    }

    fn fake_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xABu8; 36])
    }

    fn fake_hash_b() -> ActionHash {
        ActionHash::from_raw_36(vec![0xCDu8; 36])
    }

    fn fake_ts() -> Timestamp {
        Timestamp::from_micros(1_000_000)
    }

    // ========================================================================
    // 1. ALL ENUM SERDE ROUNDTRIPS (every variant)
    // ========================================================================

    // ---- HearthType (all 8 variants) ----

    #[test]
    fn hearth_type_all_variants_serde() {
        let variants = vec![
            HearthType::Nuclear,
            HearthType::Extended,
            HearthType::Chosen,
            HearthType::Blended,
            HearthType::Multigenerational,
            HearthType::Intentional,
            HearthType::CoPod,
            HearthType::Custom("Commune".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: HearthType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- MemberRole (all 7 variants) ----

    #[test]
    fn member_role_all_variants_serde() {
        let variants = vec![
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- MembershipStatus (all 4 variants) ----

    #[test]
    fn membership_status_all_variants_serde() {
        let variants = vec![
            MembershipStatus::Active,
            MembershipStatus::Invited,
            MembershipStatus::Departed,
            MembershipStatus::Ancestral,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: MembershipStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- BondType (all 13 variants) ----

    #[test]
    fn bond_type_all_variants_serde() {
        let variants = vec![
            BondType::Parent,
            BondType::Child,
            BondType::Sibling,
            BondType::Partner,
            BondType::Grandparent,
            BondType::Grandchild,
            BondType::AuntUncle,
            BondType::NieceNephew,
            BondType::Cousin,
            BondType::ChosenFamily,
            BondType::Guardian,
            BondType::Ward,
            BondType::Custom("Godparent".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: BondType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- InvitationStatus (all 4 variants) ----

    #[test]
    fn invitation_status_all_variants_serde() {
        let variants = vec![
            InvitationStatus::Pending,
            InvitationStatus::Accepted,
            InvitationStatus::Declined,
            InvitationStatus::Expired,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: InvitationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- DecisionType (all 4 variants) ----

    #[test]
    fn decision_type_all_variants_serde() {
        let variants = vec![
            DecisionType::Consensus,
            DecisionType::MajorityVote,
            DecisionType::ElderDecision,
            DecisionType::GuardianDecision,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: DecisionType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- DecisionStatus (all 3 variants) ----

    #[test]
    fn decision_status_all_variants_serde() {
        let variants = vec![
            DecisionStatus::Open,
            DecisionStatus::Closed,
            DecisionStatus::Finalized,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: DecisionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- AutonomyTier (all 5 variants) ----

    #[test]
    fn autonomy_tier_all_variants_serde() {
        let variants = vec![
            AutonomyTier::Dependent,
            AutonomyTier::Supervised,
            AutonomyTier::Guided,
            AutonomyTier::SemiAutonomous,
            AutonomyTier::Autonomous,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AutonomyTier = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- AutonomyRequestStatus (all 3 variants) ----

    #[test]
    fn autonomy_request_status_all_variants_serde() {
        let variants = vec![
            AutonomyRequestStatus::Pending,
            AutonomyRequestStatus::Approved,
            AutonomyRequestStatus::Denied,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AutonomyRequestStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- HearthVisibility (all 4 variants) ----

    #[test]
    fn hearth_visibility_all_variants_serde() {
        let variants = vec![
            HearthVisibility::AllMembers,
            HearthVisibility::AdultsOnly,
            HearthVisibility::GuardiansOnly,
            HearthVisibility::Specified(vec![fake_agent(), fake_agent_b()]),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: HearthVisibility = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn hearth_visibility_specified_empty() {
        let vis = HearthVisibility::Specified(vec![]);
        let json = serde_json::to_string(&vis).unwrap();
        let back: HearthVisibility = serde_json::from_str(&json).unwrap();
        assert_eq!(back, vis);
    }

    // ---- CareType (all 8 variants) ----

    #[test]
    fn care_type_all_variants_serde() {
        let variants = vec![
            CareType::Childcare,
            CareType::Eldercare,
            CareType::PetCare,
            CareType::Chore,
            CareType::MealPrep,
            CareType::Medical,
            CareType::Emotional,
            CareType::Custom("Tutoring".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CareType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- CareScheduleStatus (all 3 variants) ----

    #[test]
    fn care_schedule_status_all_variants_serde() {
        let variants = vec![
            CareScheduleStatus::Active,
            CareScheduleStatus::Paused,
            CareScheduleStatus::Completed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CareScheduleStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- SwapStatus (all 4 variants) ----

    #[test]
    fn swap_status_all_variants_serde() {
        let variants = vec![
            SwapStatus::Proposed,
            SwapStatus::Accepted,
            SwapStatus::Declined,
            SwapStatus::Completed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: SwapStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- Recurrence (all 4 variants) ----

    #[test]
    fn recurrence_all_variants_serde() {
        let variants = vec![
            Recurrence::Daily,
            Recurrence::Weekly,
            Recurrence::Monthly,
            Recurrence::Custom("Biweekly".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: Recurrence = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- AlertSeverity (all 4 variants) ----

    #[test]
    fn alert_severity_all_variants_serde() {
        let variants = vec![
            AlertSeverity::Low,
            AlertSeverity::Medium,
            AlertSeverity::High,
            AlertSeverity::Critical,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AlertSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- AlertType (all 6 variants) ----

    #[test]
    fn alert_type_all_variants_serde() {
        let variants = vec![
            AlertType::Medical,
            AlertType::Natural,
            AlertType::Security,
            AlertType::Missing,
            AlertType::Fire,
            AlertType::Custom("Chemical".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AlertType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- SafetyStatus (all 3 variants) ----

    #[test]
    fn safety_status_all_variants_serde() {
        let variants = vec![
            SafetyStatus::Safe,
            SafetyStatus::NeedHelp,
            SafetyStatus::NoResponse,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: SafetyStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- GratitudeType (all 5 variants) ----

    #[test]
    fn gratitude_type_all_variants_serde() {
        let variants = vec![
            GratitudeType::Appreciation,
            GratitudeType::Acknowledgment,
            GratitudeType::Celebration,
            GratitudeType::Blessing,
            GratitudeType::Custom("Thankfulness".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: GratitudeType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- CircleStatus (all 3 variants) ----

    #[test]
    fn circle_status_all_variants_serde() {
        let variants = vec![
            CircleStatus::Open,
            CircleStatus::InProgress,
            CircleStatus::Completed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CircleStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- StoryType (all 7 variants) ----

    #[test]
    fn story_type_all_variants_serde() {
        let variants = vec![
            StoryType::Memory,
            StoryType::Tradition,
            StoryType::Recipe,
            StoryType::Wisdom,
            StoryType::Origin,
            StoryType::Migration,
            StoryType::Custom("Folklore".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: StoryType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- MilestoneType (all 11 variants) ----

    #[test]
    fn milestone_type_all_variants_serde() {
        let variants = vec![
            MilestoneType::Birth,
            MilestoneType::Birthday,
            MilestoneType::FirstStep,
            MilestoneType::SchoolStart,
            MilestoneType::Graduation,
            MilestoneType::Engagement,
            MilestoneType::Marriage,
            MilestoneType::NewHome,
            MilestoneType::Retirement,
            MilestoneType::Passing,
            MilestoneType::Custom("Promotion".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: MilestoneType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- TransitionType (all 6 variants) ----

    #[test]
    fn transition_type_all_variants_serde() {
        let variants = vec![
            TransitionType::JoiningHearth,
            TransitionType::LeavingHearth,
            TransitionType::ComingOfAge,
            TransitionType::Retirement,
            TransitionType::Bereavement,
            TransitionType::Custom("Migration".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: TransitionType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- TransitionPhase (all 4 variants) ----

    #[test]
    fn transition_phase_all_variants_serde() {
        let variants = vec![
            TransitionPhase::PreLiminal,
            TransitionPhase::Liminal,
            TransitionPhase::PostLiminal,
            TransitionPhase::Integrated,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: TransitionPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- ResourceType (all 7 variants) ----

    #[test]
    fn resource_type_all_variants_serde() {
        let variants = vec![
            ResourceType::Tool,
            ResourceType::Vehicle,
            ResourceType::Book,
            ResourceType::Kitchen,
            ResourceType::Electronics,
            ResourceType::Clothing,
            ResourceType::Custom("Camping gear".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ResourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- LoanStatus (all 3 variants) ----

    #[test]
    fn loan_status_all_variants_serde() {
        let variants = vec![
            LoanStatus::Active,
            LoanStatus::Returned,
            LoanStatus::Overdue,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: LoanStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- RhythmType (all 5 variants) ----

    #[test]
    fn rhythm_type_all_variants_serde() {
        let variants = vec![
            RhythmType::Morning,
            RhythmType::Evening,
            RhythmType::Weekly,
            RhythmType::Seasonal,
            RhythmType::Custom("Annual".into()),
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: RhythmType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ---- PresenceStatusType (all 5 variants) ----

    #[test]
    fn presence_status_type_all_variants_serde() {
        let variants = vec![
            PresenceStatusType::Home,
            PresenceStatusType::Away,
            PresenceStatusType::Working,
            PresenceStatusType::Sleeping,
            PresenceStatusType::DoNotDisturb,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    // ========================================================================
    // 2. MemberRole HELPER TESTS
    // ========================================================================

    #[test]
    fn member_role_is_guardian_all_roles() {
        assert!(MemberRole::Founder.is_guardian());
        assert!(MemberRole::Elder.is_guardian());
        assert!(MemberRole::Adult.is_guardian());
        assert!(!MemberRole::Youth.is_guardian());
        assert!(!MemberRole::Child.is_guardian());
        assert!(!MemberRole::Guest.is_guardian());
        assert!(!MemberRole::Ancestor.is_guardian());
    }

    #[test]
    fn member_role_is_minor_all_roles() {
        assert!(!MemberRole::Founder.is_minor());
        assert!(!MemberRole::Elder.is_minor());
        assert!(!MemberRole::Adult.is_minor());
        assert!(MemberRole::Youth.is_minor());
        assert!(MemberRole::Child.is_minor());
        assert!(!MemberRole::Guest.is_minor());
        assert!(!MemberRole::Ancestor.is_minor());
    }

    #[test]
    fn member_role_default_vote_weight_bp_all_roles() {
        assert_eq!(MemberRole::Founder.default_vote_weight_bp(), 10000);
        assert_eq!(MemberRole::Elder.default_vote_weight_bp(), 10000);
        assert_eq!(MemberRole::Adult.default_vote_weight_bp(), 10000);
        assert_eq!(MemberRole::Youth.default_vote_weight_bp(), 5000);
        assert_eq!(MemberRole::Child.default_vote_weight_bp(), 0);
        assert_eq!(MemberRole::Guest.default_vote_weight_bp(), 0);
        assert_eq!(MemberRole::Ancestor.default_vote_weight_bp(), 0);
    }

    #[test]
    fn member_role_guardian_and_minor_mutually_exclusive() {
        // No role should be both guardian and minor
        let all_roles = vec![
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ];
        for role in &all_roles {
            assert!(
                !(role.is_guardian() && role.is_minor()),
                "{:?} is both guardian and minor",
                role
            );
        }
    }

    #[test]
    fn member_role_nonzero_vote_weight_implies_guardian_or_youth() {
        // Only guardians (Founder/Elder/Adult) and Youth have nonzero vote weight
        let all_roles = vec![
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ];
        for role in &all_roles {
            if role.default_vote_weight_bp() > 0 {
                assert!(
                    role.is_guardian() || matches!(role, MemberRole::Youth),
                    "{:?} has nonzero vote weight but is neither guardian nor Youth",
                    role
                );
            }
        }
    }

    // ========================================================================
    // 3. TransitionPhase ORDERING TESTS
    // ========================================================================

    #[test]
    fn transition_phase_forward_only_ordering() {
        assert!(TransitionPhase::PreLiminal < TransitionPhase::Liminal);
        assert!(TransitionPhase::Liminal < TransitionPhase::PostLiminal);
        assert!(TransitionPhase::PostLiminal < TransitionPhase::Integrated);
    }

    #[test]
    fn transition_phase_transitive_ordering() {
        // PreLiminal < PostLiminal (transitive)
        assert!(TransitionPhase::PreLiminal < TransitionPhase::PostLiminal);
        // PreLiminal < Integrated (transitive)
        assert!(TransitionPhase::PreLiminal < TransitionPhase::Integrated);
        // Liminal < Integrated (transitive)
        assert!(TransitionPhase::Liminal < TransitionPhase::Integrated);
    }

    #[test]
    fn transition_phase_equality() {
        assert_eq!(TransitionPhase::PreLiminal, TransitionPhase::PreLiminal);
        assert_eq!(TransitionPhase::Liminal, TransitionPhase::Liminal);
        assert_eq!(TransitionPhase::PostLiminal, TransitionPhase::PostLiminal);
        assert_eq!(TransitionPhase::Integrated, TransitionPhase::Integrated);
    }

    #[test]
    fn transition_phase_all_distinct() {
        let phases = vec![
            TransitionPhase::PreLiminal,
            TransitionPhase::Liminal,
            TransitionPhase::PostLiminal,
            TransitionPhase::Integrated,
        ];
        for (i, a) in phases.iter().enumerate() {
            for (j, b) in phases.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ========================================================================
    // 4. HearthSignal VARIANT SERDE TESTS (all 21 variants)
    // ========================================================================

    #[test]
    fn signal_gratitude_expressed_serde() {
        let sig = HearthSignal::GratitudeExpressed {
            from_agent: fake_agent(),
            to_agent: fake_agent_b(),
            message: "Thank you!".into(),
            gratitude_type: GratitudeType::Appreciation,
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::GratitudeExpressed {
                message,
                gratitude_type,
                ..
            } => {
                assert_eq!(message, "Thank you!");
                assert_eq!(gratitude_type, GratitudeType::Appreciation);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_care_task_completed_serde() {
        let sig = HearthSignal::CareTaskCompleted {
            assignee: fake_agent(),
            schedule_hash: fake_hash(),
            care_type: CareType::MealPrep,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("CareTaskCompleted"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::CareTaskCompleted { care_type, .. } => {
                assert_eq!(care_type, CareType::MealPrep);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_rhythm_occurred_serde() {
        let sig = HearthSignal::RhythmOccurred {
            rhythm_hash: fake_hash(),
            participants: vec![fake_agent(), fake_agent_b()],
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("RhythmOccurred"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::RhythmOccurred { participants, .. } => {
                assert_eq!(participants.len(), 2);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_emergency_alert_serde() {
        let sig = HearthSignal::EmergencyAlert {
            alert_hash: fake_hash(),
            severity: AlertSeverity::High,
            message: "Flood warning".into(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("EmergencyAlert"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::EmergencyAlert {
                severity, message, ..
            } => {
                assert_eq!(severity, AlertSeverity::High);
                assert_eq!(message, "Flood warning");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_member_joined_serde() {
        let sig = HearthSignal::MemberJoined {
            hearth_hash: fake_hash(),
            agent: fake_agent(),
            role: MemberRole::Adult,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("MemberJoined"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::MemberJoined { role, .. } => {
                assert_eq!(role, MemberRole::Adult);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_member_departed_serde() {
        let sig = HearthSignal::MemberDeparted {
            hearth_hash: fake_hash(),
            agent: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("MemberDeparted"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::MemberDeparted {
                hearth_hash, agent, ..
            } => {
                assert_eq!(hearth_hash, fake_hash());
                assert_eq!(agent, fake_agent());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_bond_tended_serde() {
        let sig = HearthSignal::BondTended {
            member_a: fake_agent(),
            member_b: fake_agent_b(),
            quality_bp: 8000,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("BondTended"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::BondTended { quality_bp, .. } => {
                assert_eq!(quality_bp, 8000);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_presence_changed_serde() {
        let sig = HearthSignal::PresenceChanged {
            agent: fake_agent(),
            status: PresenceStatusType::Working,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("PresenceChanged"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::PresenceChanged { status, .. } => {
                assert_eq!(status, PresenceStatusType::Working);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_cross_zome_call_failed_serde() {
        let sig = HearthSignal::CrossZomeCallFailed {
            zome: "recovery".into(),
            function: "setup_recovery".into(),
            error: "network timeout".into(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("CrossZomeCallFailed"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::CrossZomeCallFailed {
                zome,
                function,
                error,
            } => {
                assert_eq!(zome, "recovery");
                assert_eq!(function, "setup_recovery");
                assert_eq!(error, "network timeout");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_vote_cast_serde() {
        let sig = HearthSignal::VoteCast {
            decision_hash: fake_hash(),
            voter: fake_agent(),
            choice: 2,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("VoteCast"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::VoteCast { choice, .. } => assert_eq!(choice, 2),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_vote_amended_serde() {
        let sig = HearthSignal::VoteAmended {
            decision_hash: fake_hash(),
            voter: fake_agent(),
            old_choice: 0,
            new_choice: 1,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("VoteAmended"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::VoteAmended {
                old_choice,
                new_choice,
                ..
            } => {
                assert_eq!(old_choice, 0);
                assert_eq!(new_choice, 1);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_decision_closed_serde() {
        let sig = HearthSignal::DecisionClosed {
            decision_hash: fake_hash(),
            closed_by: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("DecisionClosed"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::DecisionClosed {
                decision_hash,
                closed_by,
            } => {
                assert_eq!(decision_hash, fake_hash());
                assert_eq!(closed_by, fake_agent());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_decision_finalized_serde() {
        let sig = HearthSignal::DecisionFinalized {
            decision_hash: fake_hash(),
            chosen_option: 1,
            participation_rate_bp: 8500,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("DecisionFinalized"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::DecisionFinalized {
                chosen_option,
                participation_rate_bp,
                ..
            } => {
                assert_eq!(chosen_option, 1);
                assert_eq!(participation_rate_bp, 8500);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_story_created_serde() {
        let sig = HearthSignal::StoryCreated {
            hearth_hash: fake_hash(),
            story_hash: fake_hash_b(),
            storyteller: fake_agent(),
            story_type: StoryType::Origin,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("StoryCreated"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::StoryCreated {
                story_type,
                storyteller,
                ..
            } => {
                assert_eq!(story_type, StoryType::Origin);
                assert_eq!(storyteller, fake_agent());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_story_updated_serde() {
        let sig = HearthSignal::StoryUpdated {
            story_hash: fake_hash(),
            updated_by: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("StoryUpdated"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::StoryUpdated {
                story_hash,
                updated_by,
            } => {
                assert_eq!(story_hash, fake_hash());
                assert_eq!(updated_by, fake_agent());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_tradition_observed_serde() {
        let sig = HearthSignal::TraditionObserved {
            tradition_hash: fake_hash(),
            observed_by: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("TraditionObserved"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::TraditionObserved {
                tradition_hash,
                observed_by,
            } => {
                assert_eq!(tradition_hash, fake_hash());
                assert_eq!(observed_by, fake_agent());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_resource_lent_serde() {
        let sig = HearthSignal::ResourceLent {
            resource_hash: fake_hash(),
            borrower: fake_agent(),
            due_date: fake_ts(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("ResourceLent"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::ResourceLent {
                borrower, due_date, ..
            } => {
                assert_eq!(borrower, fake_agent());
                assert_eq!(due_date, fake_ts());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_resource_returned_serde() {
        let sig = HearthSignal::ResourceReturned {
            loan_hash: fake_hash(),
            borrower: fake_agent(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("ResourceReturned"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::ResourceReturned {
                loan_hash,
                borrower,
            } => {
                assert_eq!(loan_hash, fake_hash());
                assert_eq!(borrower, fake_agent());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_expense_logged_serde() {
        let sig = HearthSignal::ExpenseLogged {
            budget_hash: fake_hash(),
            amount_cents: 4299,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("ExpenseLogged"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::ExpenseLogged {
                budget_hash,
                amount_cents,
            } => {
                assert_eq!(budget_hash, fake_hash());
                assert_eq!(amount_cents, 4299);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_milestone_recorded_serde() {
        let sig = HearthSignal::MilestoneRecorded {
            hearth_hash: fake_hash(),
            milestone_hash: fake_hash_b(),
            milestone_type: MilestoneType::Marriage,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("MilestoneRecorded"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::MilestoneRecorded { milestone_type, .. } => {
                assert_eq!(milestone_type, MilestoneType::Marriage);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_transition_advanced_serde() {
        let sig = HearthSignal::TransitionAdvanced {
            transition_hash: fake_hash(),
            new_phase: TransitionPhase::PostLiminal,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("TransitionAdvanced"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::TransitionAdvanced { new_phase, .. } => {
                assert_eq!(new_phase, TransitionPhase::PostLiminal);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_tier_advanced_serde() {
        let sig = HearthSignal::TierAdvanced {
            profile_hash: fake_hash(),
            from_tier: AutonomyTier::Supervised,
            to_tier: AutonomyTier::Guided,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("TierAdvanced"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::TierAdvanced {
                from_tier, to_tier, ..
            } => {
                assert_eq!(from_tier, AutonomyTier::Supervised);
                assert_eq!(to_tier, AutonomyTier::Guided);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_capability_approved_serde() {
        let sig = HearthSignal::CapabilityApproved {
            request_hash: fake_hash(),
            capability: "use_stove".into(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("CapabilityApproved"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::CapabilityApproved {
                request_hash,
                capability,
            } => {
                assert_eq!(request_hash, fake_hash());
                assert_eq!(capability, "use_stove");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_capability_denied_serde() {
        let sig = HearthSignal::CapabilityDenied {
            request_hash: fake_hash(),
            capability: "drive_car".into(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("CapabilityDenied"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::CapabilityDenied {
                request_hash,
                capability,
            } => {
                assert_eq!(request_hash, fake_hash());
                assert_eq!(capability, "drive_car");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_transition_progressed_serde() {
        let sig = HearthSignal::TransitionProgressed {
            transition_hash: fake_hash(),
            new_phase: TransitionPhase::Liminal,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("TransitionProgressed"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::TransitionProgressed {
                transition_hash,
                new_phase,
            } => {
                assert_eq!(transition_hash, fake_hash());
                assert_eq!(new_phase, TransitionPhase::Liminal);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_swap_accepted_serde() {
        let sig = HearthSignal::SwapAccepted {
            swap_hash: fake_hash(),
            hearth_hash: fake_hash_b(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("SwapAccepted"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::SwapAccepted {
                swap_hash,
                hearth_hash,
            } => {
                assert_eq!(swap_hash, fake_hash());
                assert_eq!(hearth_hash, fake_hash_b());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn signal_swap_declined_serde() {
        let sig = HearthSignal::SwapDeclined {
            swap_hash: fake_hash(),
            hearth_hash: fake_hash_b(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("SwapDeclined"));
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::SwapDeclined {
                swap_hash,
                hearth_hash,
            } => {
                assert_eq!(swap_hash, fake_hash());
                assert_eq!(hearth_hash, fake_hash_b());
            }
            _ => panic!("Wrong variant"),
        }
    }

    // ========================================================================
    // 5. DISTINCTNESS TESTS (all variants in each enum are unique)
    // ========================================================================

    #[test]
    fn all_hearth_types_distinct() {
        let types = vec![
            HearthType::Nuclear,
            HearthType::Extended,
            HearthType::Chosen,
            HearthType::Blended,
            HearthType::Multigenerational,
            HearthType::Intentional,
            HearthType::CoPod,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn all_bond_types_distinct() {
        let types = vec![
            BondType::Parent,
            BondType::Child,
            BondType::Sibling,
            BondType::Partner,
            BondType::Grandparent,
            BondType::Grandchild,
            BondType::AuntUncle,
            BondType::NieceNephew,
            BondType::Cousin,
            BondType::ChosenFamily,
            BondType::Guardian,
            BondType::Ward,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn all_member_roles_distinct() {
        let roles = vec![
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ];
        for (i, a) in roles.iter().enumerate() {
            for (j, b) in roles.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn all_decision_types_distinct() {
        let types = vec![
            DecisionType::Consensus,
            DecisionType::MajorityVote,
            DecisionType::ElderDecision,
            DecisionType::GuardianDecision,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn all_alert_severities_distinct() {
        let severities = vec![
            AlertSeverity::Low,
            AlertSeverity::Medium,
            AlertSeverity::High,
            AlertSeverity::Critical,
        ];
        for (i, a) in severities.iter().enumerate() {
            for (j, b) in severities.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ========================================================================
    // 6. CUSTOM VARIANT STRING PRESERVATION
    // ========================================================================

    #[test]
    fn custom_variants_preserve_empty_string() {
        let empty_custom_variants: Vec<Box<dyn std::fmt::Debug>> = vec![
            Box::new(HearthType::Custom(String::new())),
            Box::new(BondType::Custom(String::new())),
            Box::new(CareType::Custom(String::new())),
            Box::new(AlertType::Custom(String::new())),
            Box::new(GratitudeType::Custom(String::new())),
            Box::new(StoryType::Custom(String::new())),
            Box::new(MilestoneType::Custom(String::new())),
            Box::new(TransitionType::Custom(String::new())),
            Box::new(ResourceType::Custom(String::new())),
            Box::new(RhythmType::Custom(String::new())),
            Box::new(Recurrence::Custom(String::new())),
        ];
        // Just verify they can be constructed and debug-printed
        for v in &empty_custom_variants {
            let debug = format!("{:?}", v);
            assert!(debug.contains("Custom"), "Missing Custom in {:?}", debug);
        }
    }

    #[test]
    fn custom_variants_preserve_unicode() {
        let ht = HearthType::Custom("大家庭".into());
        let json = serde_json::to_string(&ht).unwrap();
        let back: HearthType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ht);

        let st = StoryType::Custom("Geschichten der Großeltern".into());
        let json = serde_json::to_string(&st).unwrap();
        let back: StoryType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, st);
    }

    // ========================================================================
    // 7. Bond Strength / Decay (preserved from existing tests)
    // ========================================================================

    #[test]
    fn decay_zero_days_returns_initial() {
        assert_eq!(decayed_strength(7000, 0), 7000);
        assert_eq!(decayed_strength(BOND_MAX, 0), BOND_MAX);
    }

    #[test]
    fn decay_one_day() {
        // 7000 * 9802 / 10000 = 6861
        let result = decayed_strength(7000, 1);
        assert_eq!(result, 6861);
    }

    #[test]
    fn decay_30_days() {
        // 7000 * 5488 / 10000 = 3841
        let result = decayed_strength(7000, 30);
        assert_eq!(result, 3841);
    }

    #[test]
    fn decay_never_below_minimum() {
        let result = decayed_strength(7000, 365);
        assert!(result >= BOND_MIN, "got {result}, expected >= {BOND_MIN}");
    }

    #[test]
    fn decay_beyond_table_clamps() {
        let result = decayed_strength(7000, 500);
        assert!(result >= BOND_MIN);
    }

    #[test]
    fn decay_interpolation_monotonic() {
        let mut prev = decayed_strength(BOND_MAX, 0);
        for day in 1..=365 {
            let current = decayed_strength(BOND_MAX, day);
            assert!(
                current <= prev,
                "decay not monotonic at day {day}: {current} > {prev}"
            );
            prev = current;
        }
    }

    #[test]
    fn decay_zero_initial_stays_zero() {
        // A bond with 0 strength should never inflate to BOND_MIN
        assert_eq!(decayed_strength(0, 1), 0);
        assert_eq!(decayed_strength(0, 30), 0);
        assert_eq!(decayed_strength(0, 365), 0);
    }

    #[test]
    fn decay_sub_minimum_stays_unchanged() {
        // Bonds below BOND_MIN should not decay or inflate
        assert_eq!(decayed_strength(500, 1), 500);
        assert_eq!(decayed_strength(500, 30), 500);
        assert_eq!(decayed_strength(999, 365), 999);
    }

    #[test]
    fn decay_exactly_minimum_decays_normally() {
        // Bond at exactly BOND_MIN should still decay (it IS at minimum, not below)
        let result = decayed_strength(BOND_MIN, 30);
        assert_eq!(result, BOND_MIN); // 1000 * 5488 / 10000 = 548, clamped to 1000
    }

    #[test]
    fn decay_max_strength_table_exact_hits() {
        // At exact table entries with initial=BOND_MAX, result should equal table value
        // (or BOND_MIN if table value is very small)
        for &(days, expected_bp) in DECAY_TABLE {
            let result = decayed_strength(BOND_MAX, days);
            let expected = expected_bp.max(BOND_MIN);
            assert_eq!(result, expected, "mismatch at day {days}");
        }
    }

    // ========================================================================
    // 8. STRUCT SERDE ROUNDTRIPS (preserved from existing tests)
    // ========================================================================

    #[test]
    fn weekly_digest_empty_serde() {
        let digest = WeeklyDigest {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            epoch_start: Timestamp::from_micros(0),
            epoch_end: Timestamp::from_micros(604_800_000_000),
            bond_updates: vec![],
            care_summary: vec![],
            gratitude_summary: vec![],
            rhythm_summary: vec![],
            created_by: AgentPubKey::from_raw_36(vec![0u8; 36]),
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&digest).unwrap();
        let back: WeeklyDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bond_updates.len(), 0);
        assert_eq!(back.care_summary.len(), 0);
    }

    #[test]
    fn weekly_digest_with_data_serde() {
        let digest = WeeklyDigest {
            hearth_hash: fake_hash(),
            epoch_start: Timestamp::from_micros(0),
            epoch_end: Timestamp::from_micros(604_800_000_000),
            bond_updates: vec![BondUpdate {
                member_a: fake_agent(),
                member_b: fake_agent_b(),
                co_creation_count: 3,
                quality_sum_bp: 24000,
            }],
            care_summary: vec![CareSummary {
                assignee: fake_agent(),
                tasks_completed: 5,
                hours_hundredths: 1200,
            }],
            gratitude_summary: vec![GratitudeSummary {
                from_agent: fake_agent(),
                to_agent: fake_agent_b(),
                count: 2,
            }],
            rhythm_summary: vec![RhythmSummary {
                rhythm_hash: fake_hash(),
                occurrences: 7,
                avg_participation_bp: 9000,
            }],
            created_by: fake_agent(),
            created_at: fake_ts(),
        };
        let json = serde_json::to_string(&digest).unwrap();
        let back: WeeklyDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bond_updates.len(), 1);
        assert_eq!(back.bond_updates[0].co_creation_count, 3);
        assert_eq!(back.care_summary.len(), 1);
        assert_eq!(back.care_summary[0].tasks_completed, 5);
        assert_eq!(back.gratitude_summary.len(), 1);
        assert_eq!(back.gratitude_summary[0].count, 2);
        assert_eq!(back.rhythm_summary.len(), 1);
        assert_eq!(back.rhythm_summary[0].occurrences, 7);
    }

    #[test]
    fn severance_input_serde_roundtrip() {
        let input = SeveranceInput {
            hearth_hash: fake_hash(),
            member_hash: fake_hash_b(),
            export_milestones: true,
            export_care_history: true,
            export_bond_snapshot: false,
            new_role: MemberRole::Adult,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SeveranceInput = serde_json::from_str(&json).unwrap();
        assert!(back.export_milestones);
        assert!(back.export_care_history);
        assert!(!back.export_bond_snapshot);
        assert_eq!(back.new_role, MemberRole::Adult);
    }

    #[test]
    fn severance_summary_data_serde_roundtrip() {
        let summary = SeveranceSummaryData {
            hearth_hash: fake_hash(),
            member: fake_agent(),
            milestones_exported: 12,
            care_records_exported: 48,
            bond_snapshot_exported: true,
            new_role: MemberRole::Ancestor,
            completed_at: fake_ts(),
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: SeveranceSummaryData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.milestones_exported, 12);
        assert_eq!(back.care_records_exported, 48);
        assert!(back.bond_snapshot_exported);
        assert_eq!(back.new_role, MemberRole::Ancestor);
    }

    #[test]
    fn bond_update_serde_roundtrip() {
        let bu = BondUpdate {
            member_a: fake_agent(),
            member_b: fake_agent_b(),
            co_creation_count: 5,
            quality_sum_bp: 42000,
        };
        let json = serde_json::to_string(&bu).unwrap();
        let back: BondUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.co_creation_count, 5);
        assert_eq!(back.quality_sum_bp, 42000);
    }

    #[test]
    fn care_summary_serde_roundtrip() {
        let cs = CareSummary {
            assignee: fake_agent(),
            tasks_completed: 12,
            hours_hundredths: 3550,
        };
        let json = serde_json::to_string(&cs).unwrap();
        let back: CareSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tasks_completed, 12);
        assert_eq!(back.hours_hundredths, 3550);
    }

    #[test]
    fn gratitude_summary_serde_roundtrip() {
        let gs = GratitudeSummary {
            from_agent: fake_agent(),
            to_agent: fake_agent_b(),
            count: 7,
        };
        let json = serde_json::to_string(&gs).unwrap();
        let back: GratitudeSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.count, 7);
    }

    #[test]
    fn rhythm_summary_serde_roundtrip() {
        let rs = RhythmSummary {
            rhythm_hash: fake_hash(),
            occurrences: 4,
            avg_participation_bp: 8500,
        };
        let json = serde_json::to_string(&rs).unwrap();
        let back: RhythmSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.occurrences, 4);
        assert_eq!(back.avg_participation_bp, 8500);
    }

    #[test]
    fn digest_epoch_input_serde_roundtrip() {
        let input = DigestEpochInput {
            hearth_hash: fake_hash(),
            epoch_start: Timestamp::from_micros(1_000_000),
            epoch_end: Timestamp::from_micros(604_800_000_000),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: DigestEpochInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hearth_hash, input.hearth_hash);
        assert_eq!(back.epoch_start, input.epoch_start);
        assert_eq!(back.epoch_end, input.epoch_end);
    }

    // ========================================================================
    // 9. BOND CONSTANTS
    // ========================================================================

    #[test]
    fn bond_constants_consistent() {
        assert!(BOND_MIN < BOND_BASE_FAMILY);
        assert!(BOND_BASE_FAMILY < BOND_MAX);
        assert_eq!(BOND_MAX, 10000);
        assert_eq!(BOND_BASE_FAMILY, 7000);
        assert_eq!(BOND_MIN, 1000);
    }

    #[test]
    fn decay_table_monotonically_decreasing() {
        let mut prev_bp = u32::MAX;
        for &(_, bp) in DECAY_TABLE {
            assert!(
                bp <= prev_bp,
                "Decay table not monotonically decreasing: {bp} > {prev_bp}"
            );
            prev_bp = bp;
        }
    }

    #[test]
    fn decay_table_starts_at_zero_days_max() {
        assert_eq!(DECAY_TABLE[0], (0, 10000));
    }

    #[test]
    fn decay_table_ends_at_365() {
        let last = DECAY_TABLE.last().unwrap();
        assert_eq!(last.0, 365);
    }
}
