// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Gamification Integrity Zome
//!
//! Implements a comprehensive gamification system including:
//! - **Experience Points (XP)**: Earned through learning activities
//! - **Levels**: Progression tiers based on XP
//! - **Badges/Achievements**: Unlockable rewards for milestones
//! - **Streaks**: Consecutive day tracking for engagement
//! - **Leaderboards**: Competitive rankings
//!
//! ## Design Philosophy
//!
//! The gamification system is designed to:
//! 1. **Intrinsically Motivate**: Focus on mastery, not just points
//! 2. **Avoid Dark Patterns**: No manipulative mechanics
//! 3. **Celebrate Progress**: Every learner can succeed
//! 4. **Foster Community**: Collaborative achievements

use hdi::prelude::*;

// ============== XP and Leveling ==============

/// Experience points record for a learner
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearnerXp {
    /// The learner
    pub learner: AgentPubKey,
    /// Total lifetime XP
    pub total_xp: u64,
    /// Current level (calculated from XP)
    pub level: u32,
    /// XP earned today
    pub daily_xp: u32,
    /// XP earned this week
    pub weekly_xp: u32,
    /// XP earned this month
    pub monthly_xp: u32,
    /// Last activity timestamp
    pub last_activity_at: i64,
    /// When XP record was created
    pub created_at: i64,
    /// When XP record was last updated
    pub modified_at: i64,
}

/// XP transaction record
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct XpTransaction {
    /// The learner who earned/spent XP
    pub learner: AgentPubKey,
    /// Amount of XP (positive = earned, negative = spent)
    pub amount: i64,
    /// Type of activity that generated the XP
    pub activity_type: XpActivityType,
    /// Reference to the activity (course, review, etc.)
    pub reference_hash: Option<ActionHash>,
    /// When this transaction occurred
    pub occurred_at: i64,
    /// Optional description
    pub description: String,
}

/// Types of activities that can earn XP
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum XpActivityType {
    /// Completed a lesson
    LessonComplete,
    /// Completed a course
    CourseComplete,
    /// Reviewed a card (SRS)
    CardReview,
    /// Perfect review session
    PerfectSession,
    /// Achieved a streak milestone
    StreakBonus,
    /// Earned a badge
    BadgeEarned,
    /// Helped another learner
    PeerHelp,
    /// Contributed to learning content
    ContentContribution,
    /// Won a pod challenge
    ChallengeWin,
    /// Daily login bonus
    DailyLogin,
    /// First activity of the week
    WeeklyBonus,
    /// Completed a skill tree
    SkillTreeComplete,
    /// Mastered a knowledge node
    NodeMastery,
    /// Custom/admin-granted XP
    Custom,
}

impl Default for XpActivityType {
    fn default() -> Self {
        XpActivityType::Custom
    }
}

/// XP multiplier for special events or bonuses
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct XpMultiplier {
    /// Description of the multiplier
    pub name: String,
    /// Multiplier value as permille (1500 = 1.5x)
    pub multiplier_permille: u16,
    /// When the multiplier starts
    pub starts_at: i64,
    /// When the multiplier ends (None = permanent)
    pub ends_at: Option<i64>,
    /// Which activity types this applies to (empty = all)
    pub applies_to: Vec<XpActivityType>,
    /// Is this multiplier active
    pub is_active: bool,
}

// ============== Badges and Achievements ==============

/// A badge definition (template for badges that can be earned)
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct BadgeDefinition {
    /// Unique identifier for this badge type
    pub badge_id: String,
    /// Display name
    pub name: String,
    /// Description of how to earn this badge
    pub description: String,
    /// Icon/emoji for the badge
    pub icon: String,
    /// Rarity tier
    pub rarity: BadgeRarity,
    /// Category of the badge
    pub category: BadgeCategory,
    /// XP reward for earning this badge
    pub xp_reward: u32,
    /// Criteria for earning (encoded as JSON)
    pub criteria_json: String,
    /// Is this badge currently earnable
    pub is_active: bool,
    /// Is this a hidden/secret badge
    pub is_secret: bool,
    /// When this badge was created
    pub created_at: i64,
}

/// Rarity tiers for badges
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BadgeRarity {
    /// Common badges (easy to earn)
    Common,
    /// Uncommon badges (moderate effort)
    Uncommon,
    /// Rare badges (significant effort)
    Rare,
    /// Epic badges (major achievement)
    Epic,
    /// Legendary badges (exceptional accomplishment)
    Legendary,
    /// Mythic badges (once-in-a-lifetime)
    Mythic,
}

impl Default for BadgeRarity {
    fn default() -> Self {
        BadgeRarity::Common
    }
}

/// Categories for badges
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BadgeCategory {
    /// Learning progress badges
    Learning,
    /// Streak-related badges
    Streaks,
    /// Social/community badges
    Community,
    /// Challenge/competition badges
    Challenges,
    /// Content creation badges
    Creation,
    /// Special event badges
    Events,
    /// Milestone badges
    Milestones,
    /// Secret/hidden badges
    Secret,
}

impl Default for BadgeCategory {
    fn default() -> Self {
        BadgeCategory::Learning
    }
}

/// A badge earned by a learner
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct EarnedBadge {
    /// The learner who earned the badge
    pub learner: AgentPubKey,
    /// Reference to the badge definition
    pub badge_definition_hash: ActionHash,
    /// When the badge was earned
    pub earned_at: i64,
    /// Progress at time of earning (for badges with tiers)
    pub progress_at_earn: u32,
    /// Optional context (e.g., which course, which streak)
    pub context_hash: Option<ActionHash>,
    /// Is this badge displayed publicly
    pub is_displayed: bool,
}

/// Progress towards earning a badge
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct BadgeProgress {
    /// The learner
    pub learner: AgentPubKey,
    /// Reference to the badge definition
    pub badge_definition_hash: ActionHash,
    /// Current progress value
    pub current_progress: u32,
    /// Required progress to earn
    pub required_progress: u32,
    /// Progress percentage as permille (500 = 50%)
    pub progress_permille: u16,
    /// When progress was last updated
    pub updated_at: i64,
}

// ============== Streaks ==============

/// Current streak status for a learner
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearnerStreak {
    /// The learner
    pub learner: AgentPubKey,
    /// Current streak length in days
    pub current_streak: u32,
    /// Longest streak ever achieved
    pub longest_streak: u32,
    /// Total days active (not necessarily consecutive)
    pub total_active_days: u32,
    /// Date of last activity (YYYYMMDD format)
    pub last_active_date: u32,
    /// Whether streak is currently frozen (saved for later)
    pub is_frozen: bool,
    /// Number of streak freezes remaining
    pub freezes_remaining: u8,
    /// Date streak started
    pub streak_start_date: u32,
    /// XP bonus multiplier for current streak as permille
    pub streak_bonus_permille: u16,
    /// When record was created
    pub created_at: i64,
    /// When record was last updated
    pub modified_at: i64,
}

/// Daily activity record for streak tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct DailyActivity {
    /// The learner
    pub learner: AgentPubKey,
    /// Date (YYYYMMDD format)
    pub date: u32,
    /// Number of lessons completed
    pub lessons_completed: u32,
    /// Number of cards reviewed
    pub cards_reviewed: u32,
    /// Number of courses advanced
    pub courses_progressed: u32,
    /// Total XP earned this day
    pub xp_earned: u32,
    /// Time spent learning (seconds)
    pub time_spent_seconds: u32,
    /// Whether this day counts for streak (met minimum activity)
    pub counts_for_streak: bool,
    /// Activities completed
    pub activities: Vec<String>,
}

/// Streak milestone achievement
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct StreakMilestone {
    /// The learner who hit the milestone
    pub learner: AgentPubKey,
    /// Streak length at milestone
    pub streak_length: u32,
    /// XP bonus awarded
    pub xp_bonus: u32,
    /// Badge awarded (if any)
    pub badge_hash: Option<ActionHash>,
    /// When milestone was achieved
    pub achieved_at: i64,
}

// ============== Leaderboards ==============

/// Leaderboard definition
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Leaderboard {
    /// Unique identifier
    pub leaderboard_id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Type of ranking
    pub ranking_type: LeaderboardType,
    /// Time period for this leaderboard
    pub time_period: LeaderboardPeriod,
    /// Maximum entries to display
    pub max_entries: u32,
    /// Is this leaderboard active
    pub is_active: bool,
    /// Scope (global, course, pod)
    pub scope: LeaderboardScope,
    /// Reference hash for scoped leaderboards
    pub scope_hash: Option<ActionHash>,
    /// When created
    pub created_at: i64,
}

/// Types of leaderboard rankings
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaderboardType {
    /// Total XP
    TotalXp,
    /// XP earned in period
    PeriodXp,
    /// Streak length
    Streak,
    /// Badges earned
    Badges,
    /// Courses completed
    CoursesCompleted,
    /// Cards reviewed
    CardsReviewed,
    /// Accuracy/retention rate
    Accuracy,
    /// Time spent learning
    TimeSpent,
    /// Challenges won
    ChallengesWon,
    /// Peer help given
    HelpGiven,
}

impl Default for LeaderboardType {
    fn default() -> Self {
        LeaderboardType::TotalXp
    }
}

/// Time periods for leaderboards
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaderboardPeriod {
    /// All time
    AllTime,
    /// This year
    Yearly,
    /// This month
    Monthly,
    /// This week
    Weekly,
    /// Today
    Daily,
}

impl Default for LeaderboardPeriod {
    fn default() -> Self {
        LeaderboardPeriod::AllTime
    }
}

/// Scope for leaderboards
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaderboardScope {
    /// Global across all learners
    Global,
    /// Within a specific course
    Course,
    /// Within a learning pod
    Pod,
    /// Within a skill tree
    SkillTree,
    /// Friends only
    Friends,
}

impl Default for LeaderboardScope {
    fn default() -> Self {
        LeaderboardScope::Global
    }
}

/// A learner's entry on a leaderboard
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LeaderboardEntry {
    /// Reference to the leaderboard
    pub leaderboard_hash: ActionHash,
    /// The learner
    pub learner: AgentPubKey,
    /// Current rank
    pub rank: u32,
    /// Previous rank (for movement tracking)
    pub previous_rank: u32,
    /// Score/value for ranking
    pub score: u64,
    /// Change from previous period
    pub score_change: i64,
    /// Rank change (positive = moved up)
    pub rank_change: i32,
    /// When entry was last updated
    pub updated_at: i64,
}

// ============== Rewards and Unlocks ==============

/// A reward that can be claimed
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Reward {
    /// Unique identifier
    pub reward_id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Type of reward
    pub reward_type: RewardType,
    /// Cost in XP (or other currency)
    pub cost_xp: u32,
    /// Level requirement
    pub level_required: u32,
    /// Badge requirements (must have all)
    pub badges_required: Vec<ActionHash>,
    /// Is this reward still available
    pub is_available: bool,
    /// Limited quantity (0 = unlimited)
    pub quantity_limit: u32,
    /// Quantity claimed so far
    pub quantity_claimed: u32,
    /// When created
    pub created_at: i64,
}

/// Types of rewards
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardType {
    /// Cosmetic avatar item
    AvatarItem,
    /// Profile customization
    ProfileTheme,
    /// Exclusive badge
    ExclusiveBadge,
    /// XP multiplier boost
    XpBoost,
    /// Streak freeze
    StreakFreeze,
    /// Early access to content
    EarlyAccess,
    /// Custom/special reward
    Custom,
}

impl Default for RewardType {
    fn default() -> Self {
        RewardType::Custom
    }
}

/// A reward claimed by a learner
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct ClaimedReward {
    /// The learner who claimed
    pub learner: AgentPubKey,
    /// Reference to the reward
    pub reward_hash: ActionHash,
    /// XP spent
    pub xp_spent: u32,
    /// When claimed
    pub claimed_at: i64,
    /// Is reward currently active (for consumables)
    pub is_active: bool,
    /// When reward expires (if applicable)
    pub expires_at: Option<i64>,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    // XP and Leveling
    #[entry_type(required_validations = 1, visibility = "public")]
    LearnerXp(LearnerXp),
    #[entry_type(required_validations = 1, visibility = "private")]
    XpTransaction(XpTransaction),
    #[entry_type(required_validations = 3, visibility = "public")]
    XpMultiplier(XpMultiplier),

    // Badges
    #[entry_type(required_validations = 3, visibility = "public")]
    BadgeDefinition(BadgeDefinition),
    #[entry_type(required_validations = 1, visibility = "public")]
    EarnedBadge(EarnedBadge),
    #[entry_type(required_validations = 1, visibility = "private")]
    BadgeProgress(BadgeProgress),

    // Streaks
    #[entry_type(required_validations = 1, visibility = "public")]
    LearnerStreak(LearnerStreak),
    #[entry_type(required_validations = 1, visibility = "private")]
    DailyActivity(DailyActivity),
    #[entry_type(required_validations = 1, visibility = "public")]
    StreakMilestone(StreakMilestone),

    // Leaderboards
    #[entry_type(required_validations = 3, visibility = "public")]
    Leaderboard(Leaderboard),
    #[entry_type(required_validations = 1, visibility = "public")]
    LeaderboardEntry(LeaderboardEntry),

    // Rewards
    #[entry_type(required_validations = 3, visibility = "public")]
    Reward(Reward),
    #[entry_type(required_validations = 1, visibility = "private")]
    ClaimedReward(ClaimedReward),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Learner -> XP record
    LearnerToXp,
    /// Learner -> XP transactions
    LearnerToXpTransactions,
    /// Learner -> Earned badges
    LearnerToBadges,
    /// Learner -> Badge progress
    LearnerToBadgeProgress,
    /// Learner -> Streak record
    LearnerToStreak,
    /// Learner -> Daily activity records
    LearnerToDailyActivity,
    /// Learner -> Streak milestones
    LearnerToMilestones,
    /// Learner -> Claimed rewards
    LearnerToRewards,
    /// Badge definition -> Earned instances
    BadgeToEarned,
    /// Leaderboard -> Entries
    LeaderboardToEntries,
    /// All badge definitions anchor
    AllBadges,
    /// All leaderboards anchor
    AllLeaderboards,
    /// All rewards anchor
    AllRewards,
    /// Active multipliers anchor
    ActiveMultipliers,
}

// ============== XP Calculation Functions ==============

/// Calculate level from total XP
/// Uses a quadratic formula: XP = 100 * level^2
/// So level = sqrt(XP / 100)
pub fn calculate_level(total_xp: u64) -> u32 {
    let level = ((total_xp as f64 / 100.0).sqrt()) as u32;
    level.max(1) // Minimum level 1
}

/// Calculate XP required for a specific level
pub fn xp_for_level(level: u32) -> u64 {
    100 * (level as u64) * (level as u64)
}

/// Calculate XP required to reach next level
pub fn xp_to_next_level(total_xp: u64) -> u64 {
    let current_level = calculate_level(total_xp);
    let next_level_xp = xp_for_level(current_level + 1);
    next_level_xp.saturating_sub(total_xp)
}

/// Calculate progress percentage to next level (as permille)
pub fn level_progress_permille(total_xp: u64) -> u16 {
    let current_level = calculate_level(total_xp);
    let current_level_xp = xp_for_level(current_level);
    let next_level_xp = xp_for_level(current_level + 1);
    let level_range = next_level_xp - current_level_xp;
    let progress = total_xp - current_level_xp;
    ((progress * 1000) / level_range) as u16
}

/// Calculate streak bonus multiplier (as permille)
/// 1 day = 1000 (no bonus)
/// 7 days = 1100 (10% bonus)
/// 30 days = 1250 (25% bonus)
/// 100 days = 1500 (50% bonus)
/// 365 days = 2000 (100% bonus)
pub fn streak_bonus_permille(streak_days: u32) -> u16 {
    match streak_days {
        0..=6 => 1000,
        7..=13 => 1100,
        14..=29 => 1150,
        30..=59 => 1250,
        60..=99 => 1350,
        100..=179 => 1500,
        180..=364 => 1750,
        _ => 2000,
    }
}

// ============== Validation Functions ==============

/// Validate XP transaction
pub fn validate_xp_transaction(tx: &XpTransaction) -> ExternResult<ValidateCallbackResult> {
    // Amount should not be zero
    if tx.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "XP transaction amount cannot be zero".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate badge definition
pub fn validate_badge_definition(badge: &BadgeDefinition) -> ExternResult<ValidateCallbackResult> {
    // Badge ID should not be empty
    if badge.badge_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Badge ID cannot be empty".to_string(),
        ));
    }

    // Name should not be empty
    if badge.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Badge name cannot be empty".to_string(),
        ));
    }

    if badge.badge_id.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid("Badge ID too long (max 100 characters)".to_string()));
    }
    if badge.name.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Badge name too long (max 200 characters)".to_string()));
    }
    if badge.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid("Badge description too long (max 5000 characters)".to_string()));
    }
    if badge.criteria_json.len() > 10000 {
        return Ok(ValidateCallbackResult::Invalid("Badge criteria too long (max 10000 characters)".to_string()));
    }
    if badge.icon.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid("Badge icon too long (max 100 characters)".to_string()));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate leaderboard
pub fn validate_leaderboard(lb: &Leaderboard) -> ExternResult<ValidateCallbackResult> {
    // ID should not be empty
    if lb.leaderboard_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Leaderboard ID cannot be empty".to_string(),
        ));
    }

    if lb.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Leaderboard name cannot be empty".to_string()));
    }
    if lb.name.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid("Leaderboard name too long (max 200 characters)".to_string()));
    }
    if lb.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid("Leaderboard description too long (max 5000 characters)".to_string()));
    }

    // Max entries should be reasonable
    if lb.max_entries > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Leaderboard max entries cannot exceed 10000".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

pub fn validate_reward(reward: &Reward) -> ExternResult<ValidateCallbackResult> {
    if reward.reward_id.is_empty() { return Ok(ValidateCallbackResult::Invalid("Reward ID cannot be empty".to_string())); }
    if reward.reward_id.len() > 100 { return Ok(ValidateCallbackResult::Invalid("Reward ID too long (max 100 characters)".to_string())); }
    if reward.name.is_empty() { return Ok(ValidateCallbackResult::Invalid("Reward name cannot be empty".to_string())); }
    if reward.name.len() > 200 { return Ok(ValidateCallbackResult::Invalid("Reward name too long (max 200 characters)".to_string())); }
    if reward.description.len() > 5000 { return Ok(ValidateCallbackResult::Invalid("Reward description too long (max 5000 characters)".to_string())); }
    if reward.badges_required.len() > 50 { return Ok(ValidateCallbackResult::Invalid("Too many required badges (max 50)".to_string())); }
    Ok(ValidateCallbackResult::Valid)
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Validate callback dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::XpTransaction(tx) => validate_xp_transaction(&tx),
                EntryTypes::BadgeDefinition(badge) => validate_badge_definition(&badge),
                EntryTypes::Leaderboard(lb) => validate_leaderboard(&lb),
                EntryTypes::Reward(reward) => validate_reward(&reward),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::XpTransaction(tx) => validate_xp_transaction(&tx),
                EntryTypes::BadgeDefinition(badge) => validate_badge_definition(&badge),
                EntryTypes::Leaderboard(lb) => validate_leaderboard(&lb),
                EntryTypes::Reward(reward) => validate_reward(&reward),
                _ => Ok(ValidateCallbackResult::Valid),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_calculation() {
        // Level 1 at 0-99 XP
        assert_eq!(calculate_level(0), 1);
        assert_eq!(calculate_level(99), 1);

        // Level 1 at 100 XP
        assert_eq!(calculate_level(100), 1);

        // Level 2 at 400 XP
        assert_eq!(calculate_level(400), 2);

        // Level 10 at 10000 XP
        assert_eq!(calculate_level(10000), 10);

        // Level 100 at 1,000,000 XP
        assert_eq!(calculate_level(1_000_000), 100);
    }

    #[test]
    fn test_xp_for_level() {
        assert_eq!(xp_for_level(1), 100);
        assert_eq!(xp_for_level(2), 400);
        assert_eq!(xp_for_level(10), 10000);
        assert_eq!(xp_for_level(100), 1_000_000);
    }

    #[test]
    fn test_streak_bonus() {
        assert_eq!(streak_bonus_permille(0), 1000);
        assert_eq!(streak_bonus_permille(7), 1100);
        assert_eq!(streak_bonus_permille(30), 1250);
        assert_eq!(streak_bonus_permille(100), 1500);
        assert_eq!(streak_bonus_permille(365), 2000);
    }

    #[test]
    fn test_level_progress() {
        // At level 1 (100 XP), progress to level 2 (400 XP)
        // 100 XP = 0% progress through level 1
        // Level 1 range: 100 to 400 = 300 XP
        // At 100 XP, progress = (100-100)/(400-100) = 0%
        let progress = level_progress_permille(100);
        assert_eq!(progress, 0);

        // At 250 XP, halfway through level 1
        // (250-100)/(400-100) = 150/300 = 50%
        let progress = level_progress_permille(250);
        assert_eq!(progress, 500);
    }

    #[test]
    fn test_badge_valid() {
        let badge = BadgeDefinition { badge_id: "b1".to_string(), name: "Test".to_string(), description: "Test".to_string(), icon: "s".to_string(), rarity: BadgeRarity::Common, category: BadgeCategory::Learning, xp_reward: 50, criteria_json: "{}".to_string(), is_active: true, is_secret: false, created_at: 0 };
        assert!(matches!(validate_badge_definition(&badge).unwrap(), ValidateCallbackResult::Valid));
    }
    #[test]
    fn test_badge_id_too_long() {
        let badge = BadgeDefinition { badge_id: "x".repeat(101), name: "T".to_string(), description: "T".to_string(), icon: "s".to_string(), rarity: BadgeRarity::Common, category: BadgeCategory::Learning, xp_reward: 50, criteria_json: "{}".to_string(), is_active: true, is_secret: false, created_at: 0 };
        assert!(matches!(validate_badge_definition(&badge).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
    #[test]
    fn test_badge_description_too_long() {
        let badge = BadgeDefinition { badge_id: "b1".to_string(), name: "T".to_string(), description: "x".repeat(5001), icon: "s".to_string(), rarity: BadgeRarity::Common, category: BadgeCategory::Learning, xp_reward: 50, criteria_json: "{}".to_string(), is_active: true, is_secret: false, created_at: 0 };
        assert!(matches!(validate_badge_definition(&badge).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
    #[test]
    fn test_badge_criteria_too_long() {
        let badge = BadgeDefinition { badge_id: "b1".to_string(), name: "T".to_string(), description: "T".to_string(), icon: "s".to_string(), rarity: BadgeRarity::Common, category: BadgeCategory::Learning, xp_reward: 50, criteria_json: "x".repeat(10001), is_active: true, is_secret: false, created_at: 0 };
        assert!(matches!(validate_badge_definition(&badge).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
    #[test]
    fn test_leaderboard_valid() {
        let lb = Leaderboard { leaderboard_id: "lb1".to_string(), name: "Top".to_string(), description: "T".to_string(), ranking_type: LeaderboardType::TotalXp, time_period: LeaderboardPeriod::Weekly, max_entries: 100, is_active: true, scope: LeaderboardScope::Global, scope_hash: None, created_at: 0 };
        assert!(matches!(validate_leaderboard(&lb).unwrap(), ValidateCallbackResult::Valid));
    }
    #[test]
    fn test_leaderboard_name_too_long() {
        let lb = Leaderboard { leaderboard_id: "lb1".to_string(), name: "x".repeat(201), description: "T".to_string(), ranking_type: LeaderboardType::TotalXp, time_period: LeaderboardPeriod::AllTime, max_entries: 100, is_active: true, scope: LeaderboardScope::Global, scope_hash: None, created_at: 0 };
        assert!(matches!(validate_leaderboard(&lb).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
    #[test]
    fn test_reward_valid() {
        let reward = Reward { reward_id: "r1".to_string(), name: "Freeze".to_string(), description: "T".to_string(), reward_type: RewardType::StreakFreeze, cost_xp: 500, level_required: 5, badges_required: vec![], is_available: true, quantity_limit: 0, quantity_claimed: 0, created_at: 0 };
        assert!(matches!(validate_reward(&reward).unwrap(), ValidateCallbackResult::Valid));
    }
    #[test]
    fn test_reward_name_too_long() {
        let reward = Reward { reward_id: "r1".to_string(), name: "x".repeat(201), description: "T".to_string(), reward_type: RewardType::Custom, cost_xp: 100, level_required: 1, badges_required: vec![], is_available: true, quantity_limit: 0, quantity_claimed: 0, created_at: 0 };
        assert!(matches!(validate_reward(&reward).unwrap(), ValidateCallbackResult::Invalid(_)));
    }
}
