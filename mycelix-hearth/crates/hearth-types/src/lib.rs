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
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- HearthType ----

    #[test]
    fn hearth_type_serde_roundtrip() {
        let ht = HearthType::Multigenerational;
        let json = serde_json::to_string(&ht).unwrap();
        let back: HearthType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ht);
    }

    #[test]
    fn hearth_type_custom_serde() {
        let ht = HearthType::Custom("Commune".into());
        let json = serde_json::to_string(&ht).unwrap();
        let back: HearthType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ht);
    }

    // ---- MemberRole ----

    #[test]
    fn member_role_serde_roundtrip() {
        for role in &[
            MemberRole::Founder,
            MemberRole::Elder,
            MemberRole::Adult,
            MemberRole::Youth,
            MemberRole::Child,
            MemberRole::Guest,
            MemberRole::Ancestor,
        ] {
            let json = serde_json::to_string(role).unwrap();
            let back: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, role);
        }
    }

    #[test]
    fn member_role_guardian_check() {
        assert!(MemberRole::Founder.is_guardian());
        assert!(MemberRole::Elder.is_guardian());
        assert!(MemberRole::Adult.is_guardian());
        assert!(!MemberRole::Youth.is_guardian());
        assert!(!MemberRole::Child.is_guardian());
        assert!(!MemberRole::Guest.is_guardian());
        assert!(!MemberRole::Ancestor.is_guardian());
    }

    #[test]
    fn member_role_minor_check() {
        assert!(!MemberRole::Founder.is_minor());
        assert!(!MemberRole::Adult.is_minor());
        assert!(MemberRole::Youth.is_minor());
        assert!(MemberRole::Child.is_minor());
    }

    #[test]
    fn member_role_vote_weights() {
        assert_eq!(MemberRole::Founder.default_vote_weight_bp(), 10000);
        assert_eq!(MemberRole::Adult.default_vote_weight_bp(), 10000);
        assert_eq!(MemberRole::Youth.default_vote_weight_bp(), 5000);
        assert_eq!(MemberRole::Child.default_vote_weight_bp(), 0);
        assert_eq!(MemberRole::Guest.default_vote_weight_bp(), 0);
    }

    // ---- BondType ----

    #[test]
    fn bond_type_serde_roundtrip() {
        let bt = BondType::ChosenFamily;
        let json = serde_json::to_string(&bt).unwrap();
        let back: BondType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, bt);
    }

    #[test]
    fn bond_type_custom_serde() {
        let bt = BondType::Custom("Godparent".into());
        let json = serde_json::to_string(&bt).unwrap();
        let back: BondType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, bt);
    }

    // ---- AutonomyTier ----

    #[test]
    fn autonomy_tier_serde_roundtrip() {
        for tier in &[
            AutonomyTier::Dependent,
            AutonomyTier::Supervised,
            AutonomyTier::Guided,
            AutonomyTier::SemiAutonomous,
            AutonomyTier::Autonomous,
        ] {
            let json = serde_json::to_string(tier).unwrap();
            let back: AutonomyTier = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, tier);
        }
    }

    // ---- TransitionPhase ordering ----

    #[test]
    fn transition_phase_forward_only() {
        assert!(TransitionPhase::PreLiminal < TransitionPhase::Liminal);
        assert!(TransitionPhase::Liminal < TransitionPhase::PostLiminal);
        assert!(TransitionPhase::PostLiminal < TransitionPhase::Integrated);
    }

    // ---- Bond Strength / Decay ----

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

    // ---- Visibility ----

    #[test]
    fn visibility_serde_roundtrip() {
        let vis = HearthVisibility::AdultsOnly;
        let json = serde_json::to_string(&vis).unwrap();
        let back: HearthVisibility = serde_json::from_str(&json).unwrap();
        assert_eq!(back, vis);
    }

    // ---- AlertSeverity ----

    #[test]
    fn alert_severity_serde_roundtrip() {
        let sev = AlertSeverity::Critical;
        let json = serde_json::to_string(&sev).unwrap();
        let back: AlertSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sev);
    }

    // ---- GratitudeType ----

    #[test]
    fn gratitude_type_serde_roundtrip() {
        let gt = GratitudeType::Celebration;
        let json = serde_json::to_string(&gt).unwrap();
        let back: GratitudeType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, gt);
    }

    // ---- StoryType ----

    #[test]
    fn story_type_serde_roundtrip() {
        let st = StoryType::Recipe;
        let json = serde_json::to_string(&st).unwrap();
        let back: StoryType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, st);
    }

    // ---- MilestoneType ----

    #[test]
    fn milestone_type_serde_roundtrip() {
        let mt = MilestoneType::Graduation;
        let json = serde_json::to_string(&mt).unwrap();
        let back: MilestoneType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, mt);
    }

    // ---- TransitionType ----

    #[test]
    fn transition_type_serde_roundtrip() {
        let tt = TransitionType::ComingOfAge;
        let json = serde_json::to_string(&tt).unwrap();
        let back: TransitionType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, tt);
    }

    // ---- DecisionType ----

    #[test]
    fn decision_type_serde_roundtrip() {
        let dt = DecisionType::Consensus;
        let json = serde_json::to_string(&dt).unwrap();
        let back: DecisionType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, dt);
    }

    // ---- ResourceType ----

    #[test]
    fn resource_type_serde_roundtrip() {
        let rt = ResourceType::Vehicle;
        let json = serde_json::to_string(&rt).unwrap();
        let back: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rt);
    }

    // ---- CareType ----

    #[test]
    fn care_type_custom_serde() {
        let ct = CareType::Custom("Tutoring".into());
        let json = serde_json::to_string(&ct).unwrap();
        let back: CareType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ct);
    }

    // ---- Recurrence ----

    #[test]
    fn recurrence_serde_roundtrip() {
        let r = Recurrence::Weekly;
        let json = serde_json::to_string(&r).unwrap();
        let back: Recurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    // ---- WeeklyDigest ----

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

    // ---- SeveranceInput ----

    #[test]
    fn severance_input_serde_roundtrip() {
        let input = SeveranceInput {
            hearth_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            member_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            export_milestones: true,
            export_care_history: true,
            export_bond_snapshot: false,
            new_role: MemberRole::Adult,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SeveranceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.export_milestones, true);
        assert_eq!(back.new_role, MemberRole::Adult);
    }

    // ---- Signal types ----

    #[test]
    fn hearth_signal_gratitude_serde() {
        let sig = HearthSignal::GratitudeExpressed {
            from_agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            to_agent: AgentPubKey::from_raw_36(vec![1u8; 36]),
            message: "Thank you!".into(),
            gratitude_type: GratitudeType::Appreciation,
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("GratitudeExpressed"));
    }

    #[test]
    fn hearth_signal_emergency_serde() {
        let sig = HearthSignal::EmergencyAlert {
            alert_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            severity: AlertSeverity::Critical,
            message: "Fire alarm!".into(),
        };
        let json = serde_json::to_string(&sig).unwrap();
        assert!(json.contains("Critical"));
    }

    // ---- All types distinct ----

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
    fn rhythm_type_serde_roundtrip() {
        let rt = RhythmType::Evening;
        let json = serde_json::to_string(&rt).unwrap();
        let back: RhythmType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rt);
    }

    #[test]
    fn presence_status_serde_roundtrip() {
        let ps = PresenceStatusType::DoNotDisturb;
        let json = serde_json::to_string(&ps).unwrap();
        let back: PresenceStatusType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ps);
    }
}
