// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Integration Integrity Zome
//!
//! Defines types for cross-zome coordination in EduNet:
//!
//! - **Learning Events**: Unified events from SRS, Gamification, Adaptive Learning
//! - **Progress Aggregates**: Combined learner progress across all systems
//! - **Achievement Triggers**: Conditions that trigger badges/rewards
//! - **Session Orchestration**: Coordinated learning session management
//!
//! NOTE: This zome does NOT depend on other integrity zomes (they conflict with HDI symbols).
//! Cross-zome types are defined locally; inter-zome communication uses HDK `call` at coordinator level.

use hdi::prelude::*;

// ============== Locally-Defined Cross-Zome Types ==============
// These mirror types from other zomes for use in integration entries.
// The coordinator uses `call` to interact with actual zome data.

/// Learning style categories (mirrors adaptive_integrity::LearningStyle)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningStyle {
    Visual,
    Auditory,
    ReadingWriting,
    Kinesthetic,
    Multimodal,
}

impl Default for LearningStyle {
    fn default() -> Self {
        LearningStyle::Multimodal
    }
}

/// Mastery levels (mirrors adaptive_integrity::MasteryLevel)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasteryLevel {
    Novice,
    Beginner,
    Competent,
    Proficient,
    Expert,
    Master,
}

impl Default for MasteryLevel {
    fn default() -> Self {
        MasteryLevel::Novice
    }
}

/// Recommendation types (mirrors adaptive_integrity::RecommendationType)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    NewContent,
    Review,
    Practice,
    Challenge,
    Remedial,
    Enrichment,
}

impl Default for RecommendationType {
    fn default() -> Self {
        RecommendationType::NewContent
    }
}

/// Recommendation reasons (mirrors adaptive_integrity::RecommendationReason)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationReason {
    InZpd,
    DueForReview,
    StrengthensWeakness,
    BuildsOnStrength,
    LearnerRequested,
    PrerequisiteMet,
    TrendingPopular,
}

impl Default for RecommendationReason {
    fn default() -> Self {
        RecommendationReason::InZpd
    }
}

// ============== Learning Events ==============

/// A unified learning event that spans all systems
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearningEvent {
    /// The learner
    pub learner: AgentPubKey,
    /// Event type
    pub event_type: LearningEventType,
    /// Reference to source action (SRS card, course progress, etc.)
    pub source_hash: ActionHash,
    /// Source zome
    pub source_zome: SourceZome,

    // === Impact Metrics ===
    /// XP gained from this event
    pub xp_gained: u32,
    /// Mastery change (permille, can be negative)
    pub mastery_change: i16,
    /// Skill hashes affected
    pub skills_affected: Vec<ActionHash>,

    // === Context ===
    /// Quality of performance (0-1000)
    pub quality_permille: u16,
    /// Time spent (seconds)
    pub duration_seconds: u32,
    /// Was this part of a streak?
    pub streak_day: Option<u32>,

    /// When event occurred
    pub occurred_at: i64,
}

/// Types of learning events
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningEventType {
    /// SRS review completed
    SrsReview,
    /// SRS card graduated
    SrsGraduated,
    /// Lesson completed
    LessonComplete,
    /// Quiz passed
    QuizPassed,
    /// Quiz failed
    QuizFailed,
    /// Project submitted
    ProjectSubmit,
    /// Peer helped
    PeerHelp,
    /// Content created
    ContentCreated,
    /// Badge earned
    BadgeEarned,
    /// Skill mastered
    SkillMastered,
    /// Goal achieved
    GoalAchieved,
    /// Streak milestone
    StreakMilestone,
    /// Challenge completed
    ChallengeComplete,
    /// Daily login
    DailyLogin,
}

impl Default for LearningEventType {
    fn default() -> Self {
        LearningEventType::LessonComplete
    }
}

/// Source zome for an event
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceZome {
    Srs,
    Gamification,
    Adaptive,
    Learning,
    Credential,
    Dao,
    Pods,
    Knowledge,
}

impl Default for SourceZome {
    fn default() -> Self {
        SourceZome::Learning
    }
}

// ============== Progress Aggregates ==============

/// Combined learner progress across all systems
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearnerProgressAggregate {
    /// The learner
    pub learner: AgentPubKey,

    // === XP & Level (from Gamification) ===
    pub total_xp: u64,
    pub level: u32,
    pub xp_to_next_level: u32,

    // === Streaks (from Gamification) ===
    pub current_streak_days: u32,
    pub longest_streak_days: u32,
    pub streak_bonus_permille: u16,

    // === SRS Stats (from SRS) ===
    pub srs_cards_total: u32,
    pub srs_cards_mature: u32,
    pub srs_cards_learning: u32,
    pub srs_reviews_today: u32,
    pub srs_accuracy_permille: u16,

    // === Mastery (from Adaptive) ===
    pub skills_tracked: u32,
    pub skills_mastered: u32,
    pub avg_mastery_permille: u16,
    pub skills_due_review: u32,

    // === Learning Style (from Adaptive) ===
    pub primary_style: LearningStyle,
    pub style_confidence_permille: u16,

    // === Goals (from Adaptive) ===
    pub active_goals: u32,
    pub completed_goals: u32,
    pub goal_progress_permille: u16,

    // === Engagement ===
    pub total_sessions: u32,
    pub total_learning_minutes: u32,
    pub avg_session_minutes: u16,
    pub days_active: u32,

    // === Achievements (from Gamification) ===
    pub badges_earned: u32,
    pub rare_badges: u32,
    pub leaderboard_rank: Option<u32>,

    // === Metadata ===
    /// Confidence in aggregate (based on data points)
    pub confidence_permille: u16,
    pub last_activity_at: i64,
    pub aggregated_at: i64,
}

// ============== Achievement Triggers ==============

/// Conditions that trigger achievements
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct AchievementTrigger {
    /// Unique trigger ID
    pub trigger_id: String,
    /// Human-readable name
    pub name: String,
    /// Description of achievement
    pub description: String,

    /// Conditions to meet (all must be satisfied)
    pub conditions: Vec<TriggerCondition>,

    /// Reward when triggered
    pub reward: TriggerReward,

    /// Can this be earned multiple times?
    pub repeatable: bool,
    /// Cooldown between repeats (seconds)
    pub cooldown_seconds: Option<u32>,

    /// Is this trigger active?
    pub is_active: bool,
    pub created_at: i64,
}

/// A single condition for an achievement trigger
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerCondition {
    /// What metric to check
    pub metric: TriggerMetric,
    /// Comparison operator
    pub operator: ComparisonOp,
    /// Target value
    pub target_value: i64,
    /// Timeframe for the condition (seconds, None = all time)
    pub timeframe_seconds: Option<u32>,
}

/// Metrics that can trigger achievements
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerMetric {
    // XP & Level
    TotalXp,
    Level,
    DailyXp,

    // Streaks
    CurrentStreak,
    LongestStreak,

    // SRS
    SrsCardsReviewed,
    SrsCardsMatured,
    SrsAccuracy,
    SrsStreakPerfect,

    // Mastery
    SkillsMastered,
    AvgMastery,
    MasteryGained,

    // Goals
    GoalsCompleted,
    GoalProgress,

    // Sessions
    TotalSessions,
    TotalMinutes,
    DaysActive,

    // Social
    PeersHelped,
    ContentCreated,
    LikesReceived,

    // Special
    BadgesEarned,
    LeaderboardRank,
}

/// Comparison operators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
    Equal,
    NotEqual,
}

/// Reward for triggering an achievement
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerReward {
    /// XP to award
    pub xp_amount: u32,
    /// Badge to award (if any)
    pub badge_id: Option<String>,
    /// Custom reward data
    pub custom_data: Option<String>,
}

// ============== Session Orchestration ==============

/// An orchestrated learning session
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct OrchestratedSession {
    /// The learner
    pub learner: AgentPubKey,
    /// Session ID
    pub session_id: String,

    /// Planned activities
    pub planned_activities: Vec<PlannedActivity>,
    /// Activities completed
    pub completed_activities: Vec<CompletedActivity>,

    /// Session state
    pub state: SessionState,

    /// Target duration (minutes)
    pub target_minutes: u16,
    /// Actual duration so far (seconds)
    pub actual_seconds: u32,

    /// XP earned in session
    pub xp_earned: u32,
    /// Mastery changes
    pub mastery_changes: Vec<MasteryChange>,

    /// Flow state tracking
    pub flow_score_permille: u16,
    pub breaks_taken: u8,

    pub started_at: i64,
    pub ended_at: Option<i64>,
}

/// A planned activity in an orchestrated session
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedActivity {
    pub activity_type: PlannedActivityType,
    pub target_hash: String, // ActionHash as string
    pub estimated_minutes: u16,
    pub priority: u8,
    pub reason: String,
}

/// Types of planned activities
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlannedActivityType {
    SrsReview,
    NewLesson,
    SkillPractice,
    QuizAttempt,
    ProjectWork,
    Review,
    Challenge,
    Break,
}

impl Default for PlannedActivityType {
    fn default() -> Self {
        PlannedActivityType::SrsReview
    }
}

/// A completed activity
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedActivity {
    pub activity_type: PlannedActivityType,
    pub target_hash: String,
    pub duration_seconds: u32,
    pub success: bool,
    pub quality_permille: u16,
    pub xp_earned: u32,
    pub completed_at: i64,
}

/// Session states
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    Planning,
    Active,
    Paused,
    Completed,
    Abandoned,
}

impl Default for SessionState {
    fn default() -> Self {
        SessionState::Planning
    }
}

/// A mastery change during session
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MasteryChange {
    pub skill_hash: String,
    pub old_permille: u16,
    pub new_permille: u16,
}

// ============== Daily Summary ==============

/// Daily learning summary
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct DailyLearningReport {
    /// The learner
    pub learner: AgentPubKey,
    /// Date (YYYYMMDD format)
    pub date: u32,

    // === Activity ===
    pub sessions_count: u32,
    pub total_minutes: u32,
    pub events_count: u32,

    // === SRS ===
    pub srs_reviews: u32,
    pub srs_new_cards: u32,
    pub srs_accuracy_permille: u16,

    // === XP ===
    pub xp_earned: u32,
    pub level_ups: u8,

    // === Mastery ===
    pub skills_practiced: u32,
    pub skills_improved: u32,
    pub mastery_gained: i16,

    // === Streak ===
    pub streak_continued: bool,
    pub streak_day: u32,

    // === Goals ===
    pub goal_progress: i16,
    pub goals_completed: u8,

    // === Badges ===
    pub badges_earned: Vec<String>,

    pub generated_at: i64,
}

// ============== Graduation ==============

/// A graduation record — proof that a learner completed a pathway
/// with mastery, sovereignty, and verified credentials.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GraduationRecord {
    pub agent: AgentPubKey,
    pub domain: String,
    pub pol_score_permille: u16,
    pub sovereignty_permille: u16,
    pub active_credentials: u32,
    pub pathway_completion_permille: u16,
    pub graduated_at: i64,
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 1, visibility = "private")]
    LearningEvent(LearningEvent),
    #[entry_type(required_validations = 1, visibility = "private")]
    LearnerProgressAggregate(LearnerProgressAggregate),
    #[entry_type(required_validations = 3, visibility = "public")]
    AchievementTrigger(AchievementTrigger),
    #[entry_type(required_validations = 1, visibility = "private")]
    OrchestratedSession(OrchestratedSession),
    #[entry_type(required_validations = 1, visibility = "private")]
    DailyLearningReport(DailyLearningReport),
    #[entry_type(required_validations = 1, visibility = "public")]
    GraduationRecord(GraduationRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Learner -> Events
    LearnerToEvents,
    /// Learner -> Progress aggregate
    LearnerToProgress,
    /// Learner -> Sessions
    LearnerToOrchestratedSessions,
    /// Learner -> Daily reports
    LearnerToDailyReports,
    /// Global triggers
    GlobalTriggers,
    /// Trigger -> Earners
    TriggerToEarners,
    /// Learner -> Graduation records
    LearnerToGraduation,
}

// ============== Helper Functions ==============

/// Calculate XP based on activity quality and type
pub fn calculate_xp(
    base_xp: u32,
    quality_permille: u16,
    streak_bonus_permille: u16,
    activity_type: &LearningEventType,
) -> u32 {
    // Quality multiplier (0.5x at 0%, 1.5x at 100%)
    let quality_mult = 500 + (quality_permille as u32 / 2);

    // Activity type multiplier
    let type_mult: u32 = match activity_type {
        LearningEventType::SrsReview => 1000,
        LearningEventType::SrsGraduated => 1500,
        LearningEventType::LessonComplete => 1000,
        LearningEventType::QuizPassed => 1200,
        LearningEventType::QuizFailed => 500,
        LearningEventType::ProjectSubmit => 2000,
        LearningEventType::PeerHelp => 1500,
        LearningEventType::ContentCreated => 2500,
        LearningEventType::BadgeEarned => 1000,
        LearningEventType::SkillMastered => 3000,
        LearningEventType::GoalAchieved => 2000,
        LearningEventType::StreakMilestone => 1500,
        LearningEventType::ChallengeComplete => 2000,
        LearningEventType::DailyLogin => 200,
    };

    // Calculate final XP
    let xp = base_xp as u64 * quality_mult as u64 * type_mult as u64 * streak_bonus_permille as u64;
    (xp / 1_000_000_000) as u32 // Normalize from permille^3
}

/// Evaluate if a condition is met
pub fn evaluate_condition(condition: &TriggerCondition, value: i64) -> bool {
    match condition.operator {
        ComparisonOp::GreaterThan => value > condition.target_value,
        ComparisonOp::GreaterOrEqual => value >= condition.target_value,
        ComparisonOp::LessThan => value < condition.target_value,
        ComparisonOp::LessOrEqual => value <= condition.target_value,
        ComparisonOp::Equal => value == condition.target_value,
        ComparisonOp::NotEqual => value != condition.target_value,
    }
}

// ============== Validation Functions ==============

/// Validate a learning event
pub fn validate_learning_event(event: &LearningEvent) -> ExternResult<ValidateCallbackResult> {
    // Quality permille must be 0-1000
    if event.quality_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quality permille must be between 0 and 1000".to_string(),
        ));
    }

    // Duration should be reasonable (max 24 hours)
    if event.duration_seconds > 86400 {
        return Ok(ValidateCallbackResult::Invalid(
            "Duration cannot exceed 24 hours (86400 seconds)".to_string(),
        ));
    }

    // Mastery change should be reasonable (-1000 to +1000)
    if event.mastery_change < -1000 || event.mastery_change > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mastery change must be between -1000 and +1000".to_string(),
        ));
    }

    // Timestamp must be positive
    if event.occurred_at <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Event timestamp must be positive".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate learner progress aggregate
pub fn validate_progress_aggregate(progress: &LearnerProgressAggregate) -> ExternResult<ValidateCallbackResult> {
    // All permille values must be 0-1000
    let permille_fields = [
        (progress.streak_bonus_permille, "streak_bonus"),
        (progress.srs_accuracy_permille, "srs_accuracy"),
        (progress.avg_mastery_permille, "avg_mastery"),
        (progress.style_confidence_permille, "style_confidence"),
        (progress.goal_progress_permille, "goal_progress"),
        (progress.confidence_permille, "confidence"),
    ];

    for (value, name) in permille_fields {
        if value > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} permille must be between 0 and 1000", name),
            ));
        }
    }

    // Skills mastered cannot exceed skills tracked
    if progress.skills_mastered > progress.skills_tracked {
        return Ok(ValidateCallbackResult::Invalid(
            "Skills mastered cannot exceed skills tracked".to_string(),
        ));
    }

    // Completed goals cannot exceed active goals + completed goals (historical)
    // This is a sanity check - completed_goals is cumulative

    // Timestamps must be positive
    if progress.last_activity_at <= 0 || progress.aggregated_at <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Timestamps must be positive".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate achievement trigger
pub fn validate_achievement_trigger(trigger: &AchievementTrigger) -> ExternResult<ValidateCallbackResult> {
    // Trigger ID must not be empty
    if trigger.trigger_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trigger ID cannot be empty".to_string(),
        ));
    }

    // Name must not be empty
    if trigger.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trigger name cannot be empty".to_string(),
        ));
    }

    // Must have at least one condition
    if trigger.conditions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trigger must have at least one condition".to_string(),
        ));
    }

    // Validate each condition
    for condition in &trigger.conditions {
        // Timeframe, if specified, must be positive
        if let Some(timeframe) = condition.timeframe_seconds {
            if timeframe == 0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Condition timeframe must be positive or None".to_string(),
                ));
            }
        }
    }

    // Timestamp must be positive
    if trigger.created_at <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Created timestamp must be positive".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate orchestrated session
pub fn validate_orchestrated_session(session: &OrchestratedSession) -> ExternResult<ValidateCallbackResult> {
    // Session ID must not be empty
    if session.session_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Session ID cannot be empty".to_string(),
        ));
    }

    // Target duration must be reasonable (max 8 hours)
    if session.target_minutes > 480 {
        return Ok(ValidateCallbackResult::Invalid(
            "Target duration cannot exceed 8 hours (480 minutes)".to_string(),
        ));
    }

    // Flow score must be valid permille
    if session.flow_score_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Flow score must be between 0 and 1000".to_string(),
        ));
    }

    // Validate completed activities
    for activity in &session.completed_activities {
        if activity.quality_permille > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Activity quality must be between 0 and 1000".to_string(),
            ));
        }
    }

    // Validate mastery changes
    for change in &session.mastery_changes {
        if change.old_permille > 1000 || change.new_permille > 1000 {
            return Ok(ValidateCallbackResult::Invalid(
                "Mastery values must be between 0 and 1000".to_string(),
            ));
        }
    }

    // Started timestamp must be positive
    if session.started_at <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Started timestamp must be positive".to_string(),
        ));
    }

    // If ended, end must be after start
    if let Some(ended) = session.ended_at {
        if ended < session.started_at {
            return Ok(ValidateCallbackResult::Invalid(
                "End timestamp cannot be before start timestamp".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate daily learning report
pub fn validate_daily_report(report: &DailyLearningReport) -> ExternResult<ValidateCallbackResult> {
    // Date must be valid YYYYMMDD format (basic check)
    if report.date < 19700101 || report.date > 99991231 {
        return Ok(ValidateCallbackResult::Invalid(
            "Date must be in YYYYMMDD format between 1970 and 9999".to_string(),
        ));
    }

    // SRS accuracy must be valid permille
    if report.srs_accuracy_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "SRS accuracy must be between 0 and 1000".to_string(),
        ));
    }

    // Mastery gained should be reasonable
    if report.mastery_gained < -10000 || report.mastery_gained > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Daily mastery change seems unreasonable".to_string(),
        ));
    }

    // Timestamp must be positive
    if report.generated_at <= 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Generated timestamp must be positive".to_string(),
        ));
    }

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
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::LearningEvent(event) => validate_learning_event(&event),
                    EntryTypes::LearnerProgressAggregate(progress) => validate_progress_aggregate(&progress),
                    EntryTypes::AchievementTrigger(trigger) => validate_achievement_trigger(&trigger),
                    EntryTypes::OrchestratedSession(session) => validate_orchestrated_session(&session),
                    EntryTypes::DailyLearningReport(report) => validate_daily_report(&report),
                    EntryTypes::GraduationRecord(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            // Other entry types (agents, capabilities) are always valid
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
    fn test_calculate_xp_base() {
        // Base case: 100 XP, 50% quality, no streak bonus, lesson complete
        let xp = calculate_xp(100, 500, 1000, &LearningEventType::LessonComplete);
        // 100 * 750 * 1000 * 1000 / 1_000_000_000 = 75
        assert_eq!(xp, 75);
    }

    #[test]
    fn test_calculate_xp_with_streak() {
        // With streak bonus: 100 XP, 100% quality, 150% streak, quiz passed
        let xp = calculate_xp(100, 1000, 1500, &LearningEventType::QuizPassed);
        // 100 * 1000 * 1200 * 1500 / 1_000_000_000 = 180
        assert_eq!(xp, 180);
    }

    #[test]
    fn test_evaluate_condition_greater_than() {
        let condition = TriggerCondition {
            metric: TriggerMetric::TotalXp,
            operator: ComparisonOp::GreaterThan,
            target_value: 1000,
            timeframe_seconds: None,
        };
        assert!(evaluate_condition(&condition, 1001));
        assert!(!evaluate_condition(&condition, 1000));
        assert!(!evaluate_condition(&condition, 999));
    }

    #[test]
    fn test_evaluate_condition_equal() {
        let condition = TriggerCondition {
            metric: TriggerMetric::Level,
            operator: ComparisonOp::Equal,
            target_value: 10,
            timeframe_seconds: None,
        };
        assert!(evaluate_condition(&condition, 10));
        assert!(!evaluate_condition(&condition, 9));
        assert!(!evaluate_condition(&condition, 11));
    }

    #[test]
    fn test_session_state_default() {
        let state = SessionState::default();
        assert_eq!(state, SessionState::Planning);
    }

    // ============== Validation Tests ==============

    #[test]
    fn test_validate_learning_event_valid() {
        let event = LearningEvent {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            event_type: LearningEventType::LessonComplete,
            source_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            source_zome: SourceZome::Learning,
            xp_gained: 100,
            mastery_change: 50,
            skills_affected: vec![],
            quality_permille: 800,
            duration_seconds: 1800,
            streak_day: Some(5),
            occurred_at: 1704067200000000, // 2024-01-01
        };
        let result = validate_learning_event(&event).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_learning_event_invalid_quality() {
        let event = LearningEvent {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            event_type: LearningEventType::LessonComplete,
            source_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            source_zome: SourceZome::Learning,
            xp_gained: 100,
            mastery_change: 50,
            skills_affected: vec![],
            quality_permille: 1500, // Invalid: > 1000
            duration_seconds: 1800,
            streak_day: None,
            occurred_at: 1704067200000000,
        };
        let result = validate_learning_event(&event).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_achievement_trigger_valid() {
        let trigger = AchievementTrigger {
            trigger_id: "first_lesson".to_string(),
            name: "First Steps".to_string(),
            description: "Complete your first lesson".to_string(),
            conditions: vec![TriggerCondition {
                metric: TriggerMetric::TotalSessions,
                operator: ComparisonOp::GreaterOrEqual,
                target_value: 1,
                timeframe_seconds: None,
            }],
            reward: TriggerReward {
                xp_amount: 100,
                badge_id: Some("beginner".to_string()),
                custom_data: None,
            },
            repeatable: false,
            cooldown_seconds: None,
            is_active: true,
            created_at: 1704067200000000,
        };
        let result = validate_achievement_trigger(&trigger).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_achievement_trigger_empty_conditions() {
        let trigger = AchievementTrigger {
            trigger_id: "invalid".to_string(),
            name: "Invalid Trigger".to_string(),
            description: "This trigger has no conditions".to_string(),
            conditions: vec![], // Invalid: empty
            reward: TriggerReward {
                xp_amount: 0,
                badge_id: None,
                custom_data: None,
            },
            repeatable: false,
            cooldown_seconds: None,
            is_active: true,
            created_at: 1704067200000000,
        };
        let result = validate_achievement_trigger(&trigger).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_daily_report_valid() {
        let report = DailyLearningReport {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            date: 20241230,
            sessions_count: 3,
            total_minutes: 90,
            events_count: 25,
            srs_reviews: 50,
            srs_new_cards: 10,
            srs_accuracy_permille: 850,
            xp_earned: 500,
            level_ups: 0,
            skills_practiced: 5,
            skills_improved: 3,
            mastery_gained: 150,
            streak_continued: true,
            streak_day: 7,
            goal_progress: 200,
            goals_completed: 1,
            badges_earned: vec!["weekly_warrior".to_string()],
            generated_at: 1704067200000000,
        };
        let result = validate_daily_report(&report).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_daily_report_invalid_date() {
        let report = DailyLearningReport {
            learner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            date: 123, // Invalid: too short
            sessions_count: 0,
            total_minutes: 0,
            events_count: 0,
            srs_reviews: 0,
            srs_new_cards: 0,
            srs_accuracy_permille: 0,
            xp_earned: 0,
            level_ups: 0,
            skills_practiced: 0,
            skills_improved: 0,
            mastery_gained: 0,
            streak_continued: false,
            streak_day: 0,
            goal_progress: 0,
            goals_completed: 0,
            badges_earned: vec![],
            generated_at: 1704067200000000,
        };
        let result = validate_daily_report(&report).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
