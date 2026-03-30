// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! WASM-safe mirror types for the Mycelix Hearth cluster.
//!
//! These types mirror `hearth-types` but use only `serde` — no HDI/HDK
//! dependencies. This allows them to compile to `wasm32-unknown-unknown`
//! for the Leptos frontend without pulling in the Holochain runtime.
//!
//! Holochain-specific types (`AgentPubKey`, `ActionHash`, `Timestamp`)
//! are replaced with `String` (base64/hex encoded) and `i64` (microseconds).

use serde::{Deserialize, Serialize};

// ============================================================================
// Hearth Types
// ============================================================================

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    Founder,
    Elder,
    Adult,
    Youth,
    Child,
    Guest,
    Ancestor,
}

impl MemberRole {
    pub fn is_guardian(&self) -> bool {
        matches!(self, Self::Founder | Self::Elder | Self::Adult)
    }

    pub fn is_minor(&self) -> bool {
        matches!(self, Self::Youth | Self::Child)
    }

    pub fn default_vote_weight_bp(&self) -> u32 {
        match self {
            Self::Founder => 10000,
            Self::Elder => 10000,
            Self::Adult => 10000,
            Self::Youth => 5000,
            Self::Child => 0,
            Self::Guest => 0,
            Self::Ancestor => 0,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Founder => "Founder",
            Self::Elder => "Elder",
            Self::Adult => "Adult",
            Self::Youth => "Youth",
            Self::Child => "Child",
            Self::Guest => "Guest",
            Self::Ancestor => "Ancestor",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MembershipStatus {
    Active,
    Invited,
    Departed,
    Ancestral,
}

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

impl BondType {
    pub fn label(&self) -> &str {
        match self {
            Self::Parent => "Parent",
            Self::Child => "Child",
            Self::Sibling => "Sibling",
            Self::Partner => "Partner",
            Self::Grandparent => "Grandparent",
            Self::Grandchild => "Grandchild",
            Self::AuntUncle => "Aunt/Uncle",
            Self::NieceNephew => "Niece/Nephew",
            Self::Cousin => "Cousin",
            Self::ChosenFamily => "Chosen Family",
            Self::Guardian => "Guardian",
            Self::Ward => "Ward",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ============================================================================
// Autonomy Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AutonomyTier {
    Dependent,
    Supervised,
    Guided,
    SemiAutonomous,
    Autonomous,
}

impl AutonomyTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dependent => "Dependent",
            Self::Supervised => "Supervised",
            Self::Guided => "Guided",
            Self::SemiAutonomous => "Semi-Autonomous",
            Self::Autonomous => "Autonomous",
        }
    }
}

// ============================================================================
// Visibility & Privacy
// ============================================================================

/// In the WASM frontend, `Specified` uses agent key strings instead of
/// `AgentPubKey` (which requires HDI).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HearthVisibility {
    AllMembers,
    AdultsOnly,
    GuardiansOnly,
    Specified(Vec<String>),
}

// ============================================================================
// Care Types
// ============================================================================

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

impl CareType {
    pub fn label(&self) -> &str {
        match self {
            Self::Childcare => "Childcare",
            Self::Eldercare => "Eldercare",
            Self::PetCare => "Pet Care",
            Self::Chore => "Chore",
            Self::MealPrep => "Meal Prep",
            Self::Medical => "Medical",
            Self::Emotional => "Emotional",
            Self::Custom(s) => s.as_str(),
        }
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AlertSeverity {
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Low => "severity-low",
            Self::Medium => "severity-medium",
            Self::High => "severity-high",
            Self::Critical => "severity-critical",
        }
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionType {
    JoiningHearth,
    LeavingHearth,
    ComingOfAge,
    Retirement,
    Bereavement,
    Custom(String),
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionType {
    Consensus,
    MajorityVote,
    ElderDecision,
    GuardianDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionStatus {
    Open,
    Closed,
    Finalized,
}

// ============================================================================
// Resource Types
// ============================================================================

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoanStatus {
    Active,
    Returned,
    Overdue,
}

// ============================================================================
// Rhythm Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhythmType {
    Morning,
    Evening,
    Weekly,
    Seasonal,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceStatusType {
    Home,
    Away,
    Working,
    Sleeping,
    DoNotDisturb,
}

impl PresenceStatusType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::Away => "Away",
            Self::Working => "Working",
            Self::Sleeping => "Sleeping",
            Self::DoNotDisturb => "Do Not Disturb",
        }
    }
}

// ============================================================================
// Status Enums
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvitationStatus {
    Pending,
    Accepted,
    Declined,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyStatus {
    Safe,
    NeedHelp,
    NoResponse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapStatus {
    Proposed,
    Accepted,
    Declined,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CareScheduleStatus {
    Active,
    Paused,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircleStatus {
    Open,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyRequestStatus {
    Pending,
    Approved,
    Denied,
}

// ============================================================================
// Bond Strength — Fixed-Point Math (identical to hearth-types)
// ============================================================================

pub type BondStrength = u32;
pub const BOND_MAX: u32 = 10000;
pub const BOND_BASE_FAMILY: u32 = 7000;
pub const BOND_MIN: u32 = 1000;

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

/// Deterministic bond decay using integer-only math.
/// Identical to `hearth-types::decayed_strength` — consensus-safe.
pub fn decayed_strength(initial_bp: u32, days_inactive: u32) -> u32 {
    if days_inactive == 0 {
        return initial_bp;
    }
    if initial_bp < BOND_MIN {
        return initial_bp;
    }

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

    if days_inactive >= 365 {
        let factor = DECAY_TABLE.last().map(|&(_, bp)| bp).unwrap_or(7);
        let result = (initial_bp as u64 * factor as u64 / BOND_MAX as u64) as u32;
        return result.max(BOND_MIN).min(initial_bp);
    }

    if lower_days == upper_days || lower_days == days_inactive {
        let result = (initial_bp as u64 * lower_bp as u64 / BOND_MAX as u64) as u32;
        return result.max(BOND_MIN).min(initial_bp);
    }

    let day_range = upper_days - lower_days;
    let bp_range = lower_bp.saturating_sub(upper_bp);
    let day_offset = days_inactive - lower_days;

    let interpolated_factor =
        lower_bp - (bp_range as u64 * day_offset as u64 / day_range as u64) as u32;
    let result = (initial_bp as u64 * interpolated_factor as u64 / BOND_MAX as u64) as u32;
    result.max(BOND_MIN).min(initial_bp)
}

// ============================================================================
// View Types (UI-facing structs with String IDs instead of Holochain hashes)
// ============================================================================

/// Hearth summary for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HearthView {
    pub hash: String,
    pub name: String,
    pub description: String,
    pub hearth_type: HearthType,
    pub created_by: String,
    pub created_at: i64,
    pub max_members: u32,
}

/// Member view for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberView {
    pub agent: String,
    pub display_name: String,
    pub role: MemberRole,
    pub status: MembershipStatus,
    pub joined_at: i64,
}

/// Bond view for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondView {
    pub hash: String,
    pub member_a: String,
    pub member_b: String,
    pub bond_type: BondType,
    pub strength_bp: u32,
    pub last_tended: i64,
    pub created_at: i64,
}

/// Decision view for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionView {
    pub hash: String,
    pub hearth_hash: String,
    pub title: String,
    pub description: String,
    pub decision_type: DecisionType,
    pub eligible_roles: Vec<MemberRole>,
    pub options: Vec<String>,
    pub deadline: i64,
    pub quorum_bp: Option<u32>,
    pub status: DecisionStatus,
    pub created_by: String,
    pub created_at: i64,
}

/// Vote view for the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteView {
    pub decision_hash: String,
    pub voter: String,
    pub choice: u32,
    pub weight_bp: u32,
    pub reasoning: Option<String>,
    pub created_at: i64,
}

/// Gratitude expression view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GratitudeExpressionView {
    pub hash: String,
    pub from_agent: String,
    pub to_agent: String,
    pub message: String,
    pub gratitude_type: GratitudeType,
    pub visibility: HearthVisibility,
    pub created_at: i64,
}

/// Care schedule view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareScheduleView {
    pub hash: String,
    pub hearth_hash: String,
    pub care_type: CareType,
    pub title: String,
    pub description: String,
    pub assigned_to: String,
    pub recurrence: Recurrence,
    pub status: CareScheduleStatus,
    pub completed_at: Option<i64>,
}

/// Emergency alert view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyAlertView {
    pub hash: String,
    pub hearth_hash: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub reporter: String,
    pub location_hint: Option<String>,
    pub created_at: i64,
    pub resolved_at: Option<i64>,
}

/// Family story view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryView {
    pub hash: String,
    pub hearth_hash: String,
    pub title: String,
    pub content: String,
    pub storyteller: String,
    pub story_type: StoryType,
    pub tags: Vec<String>,
    pub visibility: HearthVisibility,
    pub created_at: i64,
}

/// Milestone view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneView {
    pub hash: String,
    pub hearth_hash: String,
    pub member: String,
    pub milestone_type: MilestoneType,
    pub date: i64,
    pub description: String,
}

/// Shared resource view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceView {
    pub hash: String,
    pub hearth_hash: String,
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub current_holder: Option<String>,
    pub condition: String,
    pub location: String,
}

/// Rhythm view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmView {
    pub hash: String,
    pub hearth_hash: String,
    pub name: String,
    pub rhythm_type: RhythmType,
    pub description: String,
    pub participants: Vec<String>,
}

/// Autonomy profile view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyProfileView {
    pub hash: String,
    pub hearth_hash: String,
    pub member: String,
    pub current_tier: AutonomyTier,
    pub capabilities: Vec<String>,
    pub restrictions: Vec<String>,
}

/// Presence status view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceView {
    pub agent: String,
    pub status: PresenceStatusType,
    pub expected_return: Option<i64>,
    pub updated_at: i64,
}

// ============================================================================
// Signal Types (mirrors HearthSignal but with String IDs)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HearthSignal {
    GratitudeExpressed {
        from_agent: String,
        to_agent: String,
        message: String,
        gratitude_type: GratitudeType,
    },
    CareTaskCompleted {
        assignee: String,
        schedule_hash: String,
        care_type: CareType,
    },
    EmergencyAlert {
        alert_hash: String,
        severity: AlertSeverity,
        message: String,
    },
    MemberJoined {
        hearth_hash: String,
        agent: String,
        role: MemberRole,
    },
    MemberDeparted {
        hearth_hash: String,
        agent: String,
    },
    BondTended {
        member_a: String,
        member_b: String,
        quality_bp: u32,
    },
    PresenceChanged {
        agent: String,
        status: PresenceStatusType,
    },
    VoteCast {
        decision_hash: String,
        voter: String,
        choice: u32,
    },
    DecisionFinalized {
        decision_hash: String,
        chosen_option: u32,
        participation_rate_bp: u32,
    },
}

// ============================================================================
// Vote Weight Calculation
// ============================================================================

/// Compute effective vote weight from role and consciousness tier.
/// Formula: (role_weight_bp * consciousness_weight_bp) / 10000
pub fn effective_vote_weight(role: &MemberRole, consciousness_weight_bp: u32) -> u32 {
    let role_bp = role.default_vote_weight_bp();
    (role_bp as u64 * consciousness_weight_bp as u64 / 10000) as u32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_decay_zero_days() {
        assert_eq!(decayed_strength(7000, 0), 7000);
    }

    #[test]
    fn bond_decay_30_days() {
        // 7000 * 5488 / 10000 = 3841
        assert_eq!(decayed_strength(7000, 30), 3841);
    }

    #[test]
    fn bond_decay_never_below_min() {
        assert!(decayed_strength(7000, 365) >= BOND_MIN);
    }

    #[test]
    fn bond_decay_sub_min_unchanged() {
        assert_eq!(decayed_strength(500, 30), 500);
    }

    #[test]
    fn bond_decay_beyond_table() {
        let result = decayed_strength(BOND_MAX, 400);
        assert_eq!(result, BOND_MIN);
    }

    #[test]
    fn vote_weight_adult_citizen() {
        // Adult (10000) * Citizen (7500) / 10000 = 7500
        assert_eq!(effective_vote_weight(&MemberRole::Adult, 7500), 7500);
    }

    #[test]
    fn vote_weight_youth_participant() {
        // Youth (5000) * Participant (5000) / 10000 = 2500
        assert_eq!(effective_vote_weight(&MemberRole::Youth, 5000), 2500);
    }

    #[test]
    fn vote_weight_child_zero() {
        assert_eq!(effective_vote_weight(&MemberRole::Child, 10000), 0);
    }

    #[test]
    fn hearth_view_serde_roundtrip() {
        let view = HearthView {
            hash: "abc123".into(),
            name: "Our Hearth".into(),
            description: "A warm place".into(),
            hearth_type: HearthType::Chosen,
            created_by: "agent_xyz".into(),
            created_at: 1_700_000_000,
            max_members: 10,
        };
        let json = serde_json::to_string(&view).unwrap();
        let back: HearthView = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "Our Hearth");
        assert_eq!(back.hearth_type, HearthType::Chosen);
    }

    #[test]
    fn member_role_labels() {
        assert_eq!(MemberRole::Founder.label(), "Founder");
        assert_eq!(MemberRole::Ancestor.label(), "Ancestor");
    }

    #[test]
    fn all_enum_variants_serde() {
        // Spot-check a few enums for serde stability
        let json = serde_json::to_string(&CareType::Emotional).unwrap();
        let back: CareType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, CareType::Emotional);

        let json = serde_json::to_string(&TransitionPhase::Liminal).unwrap();
        let back: TransitionPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(back, TransitionPhase::Liminal);

        let json = serde_json::to_string(&AlertSeverity::Critical).unwrap();
        let back: AlertSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(back, AlertSeverity::Critical);
    }
}
