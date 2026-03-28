// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Adaptive Learning Intelligence Integrity Zome
//!
//! Implements personalized learning through ML-inspired algorithms:
//!
//! ## Core Concepts
//!
//! ### Bayesian Knowledge Tracing (BKT)
//! Models learner mastery as a hidden Markov model:
//! - P(L): Prior probability of knowing the skill
//! - P(T): Probability of learning (transitioning from unknown to known)
//! - P(G): Probability of guessing correctly
//! - P(S): Probability of slipping (incorrect despite knowing)
//!
//! ### Learning Styles (VARK Model)
//! - Visual: Learn through seeing
//! - Auditory: Learn through hearing
//! - Reading/Writing: Learn through text
//! - Kinesthetic: Learn through doing
//!
//! ### Zone of Proximal Development (ZPD)
//! Content should be challenging but achievable - the sweet spot
//! where learning is maximized.

use hdi::prelude::*;

// ============== Learning Profile ==============

/// Comprehensive learner profile for personalization
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearnerProfile {
    /// The learner
    pub learner: AgentPubKey,

    // === Learning Style (VARK) ===
    /// Visual learning preference (0-1000 permille)
    pub visual_score_permille: u16,
    /// Auditory learning preference (0-1000 permille)
    pub auditory_score_permille: u16,
    /// Reading/Writing learning preference (0-1000 permille)
    pub reading_score_permille: u16,
    /// Kinesthetic learning preference (0-1000 permille)
    pub kinesthetic_score_permille: u16,

    // === Cognitive Profile ===
    /// Preferred session length in minutes
    pub preferred_session_minutes: u16,
    /// Optimal time of day for learning (hour 0-23)
    pub optimal_hour_start: u8,
    /// End of optimal learning window
    pub optimal_hour_end: u8,
    /// Attention span estimate in minutes
    pub attention_span_minutes: u16,
    /// Preferred difficulty level (0=easy, 500=medium, 1000=hard)
    pub preferred_difficulty_permille: u16,

    // === Performance Metrics ===
    /// Overall accuracy rate (permille)
    pub accuracy_permille: u16,
    /// Learning velocity (concepts per hour * 100)
    pub velocity_centile: u16,
    /// Retention rate after 7 days (permille)
    pub retention_7d_permille: u16,
    /// Retention rate after 30 days (permille)
    pub retention_30d_permille: u16,

    // === Engagement Patterns ===
    /// Average daily active minutes
    pub avg_daily_minutes: u16,
    /// Days since last activity
    pub days_since_active: u16,
    /// Total sessions completed
    pub total_sessions: u32,
    /// Completion rate (permille)
    pub completion_rate_permille: u16,

    // === Metadata ===
    /// Profile confidence (0-1000, higher = more data)
    pub confidence_permille: u16,
    /// Number of data points used
    pub data_points: u32,
    /// When profile was created
    pub created_at: i64,
    /// When profile was last updated
    pub modified_at: i64,
}

/// Learning style assessment response
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearningStyleAssessment {
    /// The learner
    pub learner: AgentPubKey,
    /// Question ID
    pub question_id: String,
    /// Selected answer (maps to VARK category)
    pub answer: LearningStyle,
    /// Confidence in answer (1-5)
    pub confidence: u8,
    /// When answered
    pub answered_at: i64,
}

/// Learning style categories
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

// ============== Mastery Estimation (BKT) ==============

/// Skill mastery state using Bayesian Knowledge Tracing
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct SkillMastery {
    /// The learner
    pub learner: AgentPubKey,
    /// Reference to the skill/knowledge node
    pub skill_hash: ActionHash,

    // === BKT Parameters (stored as permille) ===
    /// P(L): Current mastery estimate (0-1000)
    pub mastery_permille: u16,
    /// P(T): Learning rate for this skill (0-1000)
    pub learn_rate_permille: u16,
    /// P(G): Guess probability (0-1000)
    pub guess_permille: u16,
    /// P(S): Slip probability (0-1000)
    pub slip_permille: u16,

    // === Performance History ===
    /// Total attempts on this skill
    pub total_attempts: u32,
    /// Correct attempts
    pub correct_attempts: u32,
    /// Recent attempts (last 10, as bitfield)
    pub recent_attempts_bits: u16,
    /// Number of recent attempts tracked
    pub recent_count: u8,

    // === Temporal Data ===
    /// Time since last practice (minutes)
    pub minutes_since_practice: u32,
    /// Estimated decay amount (permille)
    pub decay_estimate_permille: u16,
    /// Optimal review time (Unix timestamp)
    pub next_optimal_review: i64,

    // === Confidence ===
    /// Confidence in mastery estimate (0-1000)
    pub confidence_permille: u16,

    /// When created
    pub created_at: i64,
    /// When last updated
    pub modified_at: i64,
}

/// Mastery level thresholds
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasteryLevel {
    /// < 200 permille
    Novice,
    /// 200-400 permille
    Beginner,
    /// 400-600 permille
    Intermediate,
    /// 600-800 permille
    Proficient,
    /// 800-950 permille
    Advanced,
    /// > 950 permille
    Expert,
}

impl MasteryLevel {
    pub fn from_permille(p: u16) -> Self {
        match p {
            0..=199 => MasteryLevel::Novice,
            200..=399 => MasteryLevel::Beginner,
            400..=599 => MasteryLevel::Intermediate,
            600..=799 => MasteryLevel::Proficient,
            800..=949 => MasteryLevel::Advanced,
            _ => MasteryLevel::Expert,
        }
    }
}

// ============== Difficulty Calibration ==============

/// Content difficulty calibration based on learner performance
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct DifficultyCalibration {
    /// Reference to the content/question
    pub content_hash: ActionHash,
    /// Content type
    pub content_type: ContentType,

    // === Difficulty Estimates ===
    /// Initial/authored difficulty (0-1000)
    pub authored_difficulty_permille: u16,
    /// Calibrated difficulty based on data (0-1000)
    pub calibrated_difficulty_permille: u16,
    /// Difficulty variance (how much it varies by learner)
    pub variance_permille: u16,

    // === Performance Data ===
    /// Total attempts across all learners
    pub total_attempts: u32,
    /// Successful attempts
    pub successful_attempts: u32,
    /// Average time to complete (seconds)
    pub avg_time_seconds: u32,
    /// Standard deviation of time
    pub time_std_dev_seconds: u32,

    // === Discrimination ===
    /// How well this separates high/low performers (0-1000)
    pub discrimination_permille: u16,

    /// When calibration was updated
    pub calibrated_at: i64,
}

/// Content types for difficulty calibration
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Lesson,
    Quiz,
    Exercise,
    Project,
    Assessment,
    Challenge,
}

impl Default for ContentType {
    fn default() -> Self {
        ContentType::Lesson
    }
}

// ============== Recommendations ==============

/// A learning recommendation
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Recommendation {
    /// The learner this is for
    pub learner: AgentPubKey,
    /// What is being recommended
    pub target_hash: ActionHash,
    /// Type of recommendation
    pub rec_type: RecommendationType,

    // === Scoring ===
    /// Overall recommendation score (0-1000)
    pub score_permille: u16,
    /// Relevance to learner goals (0-1000)
    pub relevance_permille: u16,
    /// Difficulty match to learner level (0-1000)
    pub difficulty_match_permille: u16,
    /// Prerequisite readiness (0-1000)
    pub readiness_permille: u16,
    /// Recency/freshness (0-1000)
    pub freshness_permille: u16,

    // === Reasoning ===
    /// Why this is recommended
    pub reason: RecommendationReason,
    /// Human-readable explanation
    pub explanation: String,

    // === Metadata ===
    /// Priority rank (1 = highest)
    pub rank: u32,
    /// Is this recommendation still valid
    pub is_valid: bool,
    /// When recommendation was generated
    pub generated_at: i64,
    /// When recommendation expires
    pub expires_at: i64,
}

/// Types of recommendations
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Next skill to learn
    NextSkill,
    /// Review needed
    Review,
    /// Practice exercise
    Practice,
    /// Challenge to attempt
    Challenge,
    /// Course to take
    Course,
    /// Pod to join
    Pod,
    /// Content to explore
    Exploration,
}

impl Default for RecommendationType {
    fn default() -> Self {
        RecommendationType::NextSkill
    }
}

/// Reasons for recommendations
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationReason {
    /// Follows learning path
    PathContinuation,
    /// Skill is decaying
    RetentionRisk,
    /// Prerequisite for goal
    GoalPrerequisite,
    /// Popular with similar learners
    CollaborativeFiltering,
    /// Matches learning style
    StyleMatch,
    /// Optimal difficulty
    ZoneOfProximalDevelopment,
    /// Strengthens weak area
    WeaknessRemediation,
    /// Builds on strength
    StrengthExpansion,
    /// Trending/popular
    Trending,
    /// Time-sensitive
    TimeSensitive,
}

impl Default for RecommendationReason {
    fn default() -> Self {
        RecommendationReason::PathContinuation
    }
}

// ============== Learning Goals ==============

/// A learner's goal
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct LearningGoal {
    /// The learner
    pub learner: AgentPubKey,
    /// Goal description
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Goal type
    pub goal_type: GoalType,
    /// Target skills/nodes
    pub target_hashes: Vec<ActionHash>,

    // === Progress ===
    /// Current progress (0-1000)
    pub progress_permille: u16,
    /// Target completion date
    pub target_date: Option<i64>,
    /// Is goal completed
    pub is_completed: bool,

    // === Priority ===
    /// Priority level
    pub priority: GoalPriority,
    /// Estimated time to complete (hours)
    pub estimated_hours: u16,

    /// When created
    pub created_at: i64,
    /// When last updated
    pub modified_at: i64,
}

/// Types of learning goals
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalType {
    /// Master a specific skill
    SkillMastery,
    /// Complete a course
    CourseCompletion,
    /// Earn a credential
    Credential,
    /// Build a project
    Project,
    /// Explore a topic
    Exploration,
    /// Daily/weekly habit
    Habit,
    /// Custom goal
    Custom,
}

impl Default for GoalType {
    fn default() -> Self {
        GoalType::Custom
    }
}

/// Goal priority levels
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for GoalPriority {
    fn default() -> Self {
        GoalPriority::Medium
    }
}

// ============== Learning Analytics ==============

/// Learning session analytics
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct SessionAnalytics {
    /// The learner
    pub learner: AgentPubKey,
    /// Session start time
    pub started_at: i64,
    /// Session end time
    pub ended_at: i64,
    /// Duration in seconds
    pub duration_seconds: u32,

    // === Activity Metrics ===
    /// Items attempted
    pub items_attempted: u32,
    /// Items completed
    pub items_completed: u32,
    /// Correct responses
    pub correct_count: u32,
    /// Skills touched
    pub skills_touched: Vec<ActionHash>,

    // === Engagement Metrics ===
    /// Average response time (ms)
    pub avg_response_time_ms: u32,
    /// Time on task (excluding pauses)
    pub active_time_seconds: u32,
    /// Number of hints used
    pub hints_used: u32,
    /// Number of skips
    pub skips: u32,

    // === Flow State Indicators ===
    /// Estimated focus level (0-1000)
    pub focus_estimate_permille: u16,
    /// Challenge-skill balance (0-1000, 500 = optimal)
    pub flow_balance_permille: u16,
    /// Frustration indicators
    pub frustration_signals: u8,
    /// Boredom indicators
    pub boredom_signals: u8,

    // === Outcomes ===
    /// XP earned
    pub xp_earned: u32,
    /// Mastery changes (sum of permille changes)
    pub mastery_gained: i32,
    /// New skills unlocked
    pub skills_unlocked: u32,
}

/// Aggregated analytics over time
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct AggregatedAnalytics {
    /// The learner
    pub learner: AgentPubKey,
    /// Period type
    pub period: AnalyticsPeriod,
    /// Period identifier (e.g., 202401 for Jan 2024)
    pub period_id: u32,

    // === Volume ===
    /// Total sessions
    pub total_sessions: u32,
    /// Total time (minutes)
    pub total_minutes: u32,
    /// Total items completed
    pub items_completed: u32,

    // === Performance ===
    /// Average accuracy (permille)
    pub avg_accuracy_permille: u16,
    /// Skills mastered (>800 permille)
    pub skills_mastered: u32,
    /// Mastery improvement (permille)
    pub mastery_improvement_permille: i16,

    // === Patterns ===
    /// Most active hour (0-23)
    pub peak_hour: u8,
    /// Most active day (0=Sun, 6=Sat)
    pub peak_day: u8,
    /// Average session length (minutes)
    pub avg_session_minutes: u16,

    // === Engagement ===
    /// Days active in period
    pub days_active: u16,
    /// Streak at end of period
    pub streak_at_end: u32,
    /// XP earned in period
    pub xp_earned: u32,

    /// When aggregated
    pub aggregated_at: i64,
}

/// Analytics time periods
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalyticsPeriod {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Yearly,
}

impl Default for AnalyticsPeriod {
    fn default() -> Self {
        AnalyticsPeriod::Weekly
    }
}

// ============== Adaptive Path ==============

/// A personalized learning path
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct AdaptivePath {
    /// The learner
    pub learner: AgentPubKey,
    /// Path name
    pub name: String,
    /// Target goal
    pub goal_hash: Option<ActionHash>,

    // === Path Structure ===
    /// Ordered steps in the path
    pub steps: Vec<PathStep>,
    /// Current step index
    pub current_step: u32,
    /// Completed steps
    pub completed_steps: u32,

    // === Adaptation ===
    /// How much the path has been adapted
    pub adaptation_count: u32,
    /// Last adaptation reason
    pub last_adaptation_reason: String,

    // === Estimates ===
    /// Estimated total time (hours)
    pub estimated_hours: u16,
    /// Estimated completion date
    pub estimated_completion: Option<i64>,
    /// Confidence in estimates (0-1000)
    pub estimate_confidence_permille: u16,

    /// When created
    pub created_at: i64,
    /// When last updated
    pub modified_at: i64,
}

/// A step in a learning path
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathStep {
    /// Reference to content/skill
    pub target_hash: String, // ActionHash as string for serialization
    /// Step type
    pub step_type: PathStepType,
    /// Expected duration (minutes)
    pub duration_minutes: u16,
    /// Is step completed
    pub is_completed: bool,
    /// Is step optional
    pub is_optional: bool,
    /// Reason for inclusion
    pub reason: String,
}

/// Types of path steps
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathStepType {
    Learn,
    Practice,
    Review,
    Assess,
    Project,
    Rest,
}

impl Default for PathStepType {
    fn default() -> Self {
        PathStepType::Learn
    }
}

// ============== Entry and Link Type Definitions ==============

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    // Profile
    #[entry_type(required_validations = 1, visibility = "public")]
    LearnerProfile(LearnerProfile),
    #[entry_type(required_validations = 1, visibility = "private")]
    LearningStyleAssessment(LearningStyleAssessment),

    // Mastery
    #[entry_type(required_validations = 1, visibility = "private")]
    SkillMastery(SkillMastery),
    #[entry_type(required_validations = 3, visibility = "public")]
    DifficultyCalibration(DifficultyCalibration),

    // Recommendations
    #[entry_type(required_validations = 1, visibility = "private")]
    Recommendation(Recommendation),

    // Goals
    #[entry_type(required_validations = 1, visibility = "public")]
    LearningGoal(LearningGoal),

    // Analytics
    #[entry_type(required_validations = 1, visibility = "private")]
    SessionAnalytics(SessionAnalytics),
    #[entry_type(required_validations = 1, visibility = "private")]
    AggregatedAnalytics(AggregatedAnalytics),

    // Paths
    #[entry_type(required_validations = 1, visibility = "private")]
    AdaptivePath(AdaptivePath),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Learner -> Profile
    LearnerToProfile,
    /// Learner -> Style assessments
    LearnerToAssessments,
    /// Learner -> Skill masteries
    LearnerToMasteries,
    /// Learner -> Recommendations
    LearnerToRecommendations,
    /// Learner -> Goals
    LearnerToGoals,
    /// Learner -> Session analytics
    LearnerToSessions,
    /// Learner -> Aggregated analytics
    LearnerToAggregates,
    /// Learner -> Adaptive paths
    LearnerToPaths,
    /// Skill -> Mastery records (for aggregation)
    SkillToMasteries,
    /// Content -> Difficulty calibration
    ContentToCalibration,
    /// Goal -> Related recommendations
    GoalToRecommendations,
}

// ============== BKT Calculation Functions ==============

/// Update mastery using Bayesian Knowledge Tracing
/// Returns new mastery estimate (permille)
pub fn bkt_update(
    prior_mastery_permille: u16,
    correct: bool,
    learn_rate_permille: u16,
    guess_permille: u16,
    slip_permille: u16,
) -> u16 {
    let p_l = prior_mastery_permille as f64 / 1000.0;
    let p_t = learn_rate_permille as f64 / 1000.0;
    let p_g = guess_permille as f64 / 1000.0;
    let p_s = slip_permille as f64 / 1000.0;

    // P(L|evidence) using Bayes' theorem
    let posterior = if correct {
        // P(L|correct) = P(correct|L) * P(L) / P(correct)
        // P(correct|L) = 1 - P(S)
        // P(correct|~L) = P(G)
        let p_correct_given_l = 1.0 - p_s;
        let p_correct_given_not_l = p_g;
        let p_correct = p_correct_given_l * p_l + p_correct_given_not_l * (1.0 - p_l);

        if p_correct > 0.0 {
            (p_correct_given_l * p_l) / p_correct
        } else {
            p_l
        }
    } else {
        // P(L|incorrect) = P(incorrect|L) * P(L) / P(incorrect)
        // P(incorrect|L) = P(S)
        // P(incorrect|~L) = 1 - P(G)
        let p_incorrect_given_l = p_s;
        let p_incorrect_given_not_l = 1.0 - p_g;
        let p_incorrect = p_incorrect_given_l * p_l + p_incorrect_given_not_l * (1.0 - p_l);

        if p_incorrect > 0.0 {
            (p_incorrect_given_l * p_l) / p_incorrect
        } else {
            p_l
        }
    };

    // Apply learning: P(L_n) = P(L_n-1|evidence) + (1 - P(L_n-1|evidence)) * P(T)
    let updated = posterior + (1.0 - posterior) * p_t;

    // Clamp to valid range and convert back to permille
    ((updated.max(0.0).min(1.0)) * 1000.0) as u16
}

/// Calculate optimal review time based on forgetting curve
/// Returns minutes until optimal review
pub fn optimal_review_time(
    mastery_permille: u16,
    last_review_minutes_ago: u32,
    retention_target_permille: u16,
) -> u32 {
    // Using Ebbinghaus forgetting curve approximation
    // R = e^(-t/S) where S is stability
    // Stability increases with mastery

    let mastery = mastery_permille as f64 / 1000.0;

    // Higher mastery = higher stability (slower forgetting)
    // Base stability of 1440 minutes (1 day) scaled by mastery
    let stability = 1440.0 * (0.5 + mastery * 2.0);

    let target_retention = retention_target_permille as f64 / 1000.0;

    // Solve for t: target = e^(-t/S)
    // t = -S * ln(target)
    let optimal_interval = -stability * target_retention.ln();

    // Return time until review (subtract elapsed time)
    let remaining = optimal_interval - last_review_minutes_ago as f64;
    remaining.max(0.0) as u32
}

// ============== Retention Prediction Algorithm ==============
// Based on Ebbinghaus forgetting curve and FSRS (Free Spaced Repetition Scheduler)

/// Retention prediction result with detailed analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionPrediction {
    /// Current predicted retention probability (0-1000 permille)
    pub current_retention_permille: u16,
    /// Predicted retention in 1 hour (0-1000)
    pub retention_1h_permille: u16,
    /// Predicted retention in 1 day (0-1000)
    pub retention_1d_permille: u16,
    /// Predicted retention in 7 days (0-1000)
    pub retention_7d_permille: u16,
    /// Predicted retention in 30 days (0-1000)
    pub retention_30d_permille: u16,
    /// Memory stability in minutes (half-life of retention)
    pub stability_minutes: u32,
    /// Minutes until retention drops below 80%
    pub time_to_80_percent_minutes: u32,
    /// Minutes until retention drops below 50%
    pub time_to_50_percent_minutes: u32,
    /// Difficulty factor (1.0-5.0 range, scaled to 1000-5000)
    pub difficulty_factor_permille: u16,
    /// Number of successful reviews (strengthens stability)
    pub successful_reviews: u32,
    /// Retrievability score (ease of recall, 0-1000)
    pub retrievability_permille: u16,
}

/// Calculate memory stability based on review history
/// Stability represents how long a memory will last (in minutes)
/// Uses FSRS-inspired algorithm with power-law forgetting
pub fn calculate_stability(
    mastery_permille: u16,
    successful_reviews: u32,
    difficulty_permille: u16,
    last_interval_minutes: u32,
) -> f64 {
    // Base stability: starts low, grows with each successful review
    // Formula inspired by FSRS: S = S_0 * (1 + a * e^(-D/b))^n
    // Where n = number of successful reviews, D = difficulty

    let mastery = mastery_permille as f64 / 1000.0;
    let difficulty = (difficulty_permille as f64 / 1000.0).max(0.1).min(1.0);

    // Initial stability: 60 minutes for average difficulty, scaled by mastery
    let base_stability = 60.0 * (0.3 + mastery * 1.4);

    // Difficulty factor: harder items have lower stability
    // D = 1.0 (easy) -> factor = 1.5, D = 0.5 (hard) -> factor = 0.75
    let difficulty_factor = 0.5 + (1.0 - difficulty) * 1.0;

    // Growth factor per successful review: 2.0 to 2.5 depending on mastery
    let growth_factor = 2.0 + mastery * 0.5;

    // Apply exponential growth for successful reviews (capped to prevent overflow)
    let review_multiplier = if successful_reviews > 0 {
        growth_factor.powf((successful_reviews as f64).min(10.0))
    } else {
        1.0
    };

    // Last interval contributes to stability (successful longer intervals = stronger memory)
    let interval_bonus = if last_interval_minutes > 0 {
        1.0 + (last_interval_minutes as f64 / 10080.0).min(1.0) * 0.5 // Max 50% bonus for 1 week+
    } else {
        1.0
    };

    base_stability * difficulty_factor * review_multiplier * interval_bonus
}

/// Predict retention at a given time using forgetting curve
/// R(t) = e^(-t/S) where S is stability
pub fn predict_retention_at_time(
    stability_minutes: f64,
    elapsed_minutes: f64,
) -> f64 {
    if stability_minutes <= 0.0 {
        return 0.0;
    }
    (-elapsed_minutes / stability_minutes).exp().max(0.0).min(1.0)
}

/// Calculate time until retention drops to target level
/// Solve for t: target = e^(-t/S) => t = -S * ln(target)
pub fn time_to_retention(
    stability_minutes: f64,
    target_retention: f64,
) -> f64 {
    if stability_minutes <= 0.0 || target_retention <= 0.0 || target_retention >= 1.0 {
        return 0.0;
    }
    -stability_minutes * target_retention.ln()
}

/// Calculate retrievability (ease of recall based on current state)
/// Combines retention with mastery for a holistic recall score
pub fn calculate_retrievability(
    current_retention: f64,
    mastery_permille: u16,
    confidence_permille: u16,
) -> u16 {
    let mastery = mastery_permille as f64 / 1000.0;
    let confidence = confidence_permille as f64 / 1000.0;

    // Weighted combination: retention matters most, but mastery and confidence help
    let retrievability = current_retention * 0.6 + mastery * 0.25 + confidence * 0.15;

    ((retrievability * 1000.0).max(0.0).min(1000.0)) as u16
}

/// Generate comprehensive retention prediction
/// This is the main function for predicting memory retention over time
pub fn predict_retention(
    mastery_permille: u16,
    successful_reviews: u32,
    difficulty_permille: u16,
    last_review_minutes_ago: u32,
    last_interval_minutes: u32,
    confidence_permille: u16,
) -> RetentionPrediction {
    // Calculate stability
    let stability = calculate_stability(
        mastery_permille,
        successful_reviews,
        difficulty_permille,
        last_interval_minutes,
    );

    let elapsed = last_review_minutes_ago as f64;

    // Current retention
    let current_retention = predict_retention_at_time(stability, elapsed);

    // Future retention predictions
    let retention_1h = predict_retention_at_time(stability, elapsed + 60.0);
    let retention_1d = predict_retention_at_time(stability, elapsed + 1440.0);
    let retention_7d = predict_retention_at_time(stability, elapsed + 10080.0);
    let retention_30d = predict_retention_at_time(stability, elapsed + 43200.0);

    // Time to retention thresholds (from now, not from last review)
    let time_to_80 = time_to_retention(stability, 0.8);
    let time_to_50 = time_to_retention(stability, 0.5);

    // Retrievability
    let retrievability = calculate_retrievability(
        current_retention,
        mastery_permille,
        confidence_permille,
    );

    // Difficulty factor (for UI display)
    let difficulty_factor = ((1.0 + difficulty_permille as f64 / 250.0) * 1000.0) as u16;

    RetentionPrediction {
        current_retention_permille: (current_retention * 1000.0) as u16,
        retention_1h_permille: (retention_1h * 1000.0) as u16,
        retention_1d_permille: (retention_1d * 1000.0) as u16,
        retention_7d_permille: (retention_7d * 1000.0) as u16,
        retention_30d_permille: (retention_30d * 1000.0) as u16,
        stability_minutes: stability as u32,
        time_to_80_percent_minutes: (time_to_80 - elapsed).max(0.0) as u32,
        time_to_50_percent_minutes: (time_to_50 - elapsed).max(0.0) as u32,
        difficulty_factor_permille: difficulty_factor.min(5000),
        successful_reviews,
        retrievability_permille: retrievability,
    }
}

/// Batch predict retention for multiple skills
/// Useful for dashboards and analytics
pub fn predict_retention_batch(
    skills: Vec<(u16, u32, u16, u32, u32, u16)>, // (mastery, reviews, difficulty, elapsed, interval, confidence)
) -> Vec<RetentionPrediction> {
    skills
        .into_iter()
        .map(|(mastery, reviews, difficulty, elapsed, interval, confidence)| {
            predict_retention(mastery, reviews, difficulty, elapsed, interval, confidence)
        })
        .collect()
}

/// Calculate optimal review schedule for target retention
/// Returns recommended review times to maintain retention above threshold
pub fn optimal_review_schedule(
    current_stability_minutes: f64,
    target_retention_permille: u16,
    forecast_days: u32,
) -> Vec<u32> {
    let target = target_retention_permille as f64 / 1000.0;
    let mut schedule = Vec::new();
    let mut current_time: f64 = 0.0;
    let mut stability = current_stability_minutes;
    let max_minutes = forecast_days as f64 * 1440.0;

    // Generate review schedule
    while current_time < max_minutes && schedule.len() < 100 {
        // Time until retention drops to target
        let next_review = time_to_retention(stability, target);

        if next_review <= 0.0 || next_review > max_minutes - current_time {
            break;
        }

        current_time += next_review;
        schedule.push(current_time as u32);

        // After successful review, stability increases by ~2.5x
        stability *= 2.5;
    }

    schedule
}

/// Calculate difficulty match score
/// Returns permille score (500 = perfect match)
pub fn difficulty_match_score(
    learner_level_permille: u16,
    content_difficulty_permille: u16,
) -> u16 {
    // Zone of Proximal Development: content should be slightly harder
    // Optimal is content about 50-150 permille above learner level
    let learner = learner_level_permille as i32;
    let content = content_difficulty_permille as i32;
    let diff = content - learner;

    // Optimal range: 50-150 above
    // Score decreases as we deviate from this range
    let optimal_min = 50;
    let optimal_max = 150;

    if diff >= optimal_min && diff <= optimal_max {
        // Perfect ZPD range
        1000
    } else if diff < optimal_min {
        // Too easy - score based on how far below optimal
        let distance = optimal_min - diff;
        (1000 - (distance * 3).min(1000)) as u16
    } else {
        // Too hard - score based on how far above optimal
        let distance = diff - optimal_max;
        (1000 - (distance * 5).min(1000)) as u16
    }
}

// ============== Validation Functions ==============

/// Validate learner profile
pub fn validate_learner_profile(profile: &LearnerProfile) -> ExternResult<ValidateCallbackResult> {
    // Learning style scores should be reasonable
    if profile.visual_score_permille > 1000
        || profile.auditory_score_permille > 1000
        || profile.reading_score_permille > 1000
        || profile.kinesthetic_score_permille > 1000
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Learning style scores must be 0-1000".to_string(),
        ));
    }

    // Optimal hours should be valid
    if profile.optimal_hour_start >= 24 || profile.optimal_hour_end >= 24 {
        return Ok(ValidateCallbackResult::Invalid(
            "Optimal hours must be 0-23".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate skill mastery
pub fn validate_skill_mastery(mastery: &SkillMastery) -> ExternResult<ValidateCallbackResult> {
    // BKT parameters should be in valid range
    if mastery.mastery_permille > 1000
        || mastery.learn_rate_permille > 1000
        || mastery.guess_permille > 1000
        || mastery.slip_permille > 1000
    {
        return Ok(ValidateCallbackResult::Invalid(
            "BKT parameters must be 0-1000".to_string(),
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
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::LearnerProfile(profile) => validate_learner_profile(&profile),
                EntryTypes::SkillMastery(mastery) => validate_skill_mastery(&mastery),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::LearnerProfile(profile) => validate_learner_profile(&profile),
                EntryTypes::SkillMastery(mastery) => validate_skill_mastery(&mastery),
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
    fn test_bkt_update_correct() {
        // Starting mastery: 50%, learn rate: 10%, guess: 20%, slip: 10%
        let new_mastery = bkt_update(500, true, 100, 200, 100);
        // Mastery should increase after correct answer
        assert!(new_mastery > 500);
    }

    #[test]
    fn test_bkt_update_incorrect() {
        // Starting mastery: 50%, learn rate: 10%, guess: 20%, slip: 10%
        let new_mastery = bkt_update(500, false, 100, 200, 100);
        // Mastery should decrease after incorrect answer (but learning helps)
        // Actually with learning, it might still go up slightly
        // The key is it should be less than if it were correct
        let correct_mastery = bkt_update(500, true, 100, 200, 100);
        assert!(new_mastery < correct_mastery);
    }

    #[test]
    fn test_difficulty_match_optimal() {
        // Learner at 500, content at 600 (100 above = in ZPD)
        let score = difficulty_match_score(500, 600);
        assert_eq!(score, 1000); // Perfect match
    }

    #[test]
    fn test_difficulty_match_too_easy() {
        // Learner at 500, content at 400 (100 below)
        let score = difficulty_match_score(500, 400);
        assert!(score < 1000); // Not optimal
    }

    #[test]
    fn test_difficulty_match_too_hard() {
        // Learner at 500, content at 800 (300 above)
        let score = difficulty_match_score(500, 800);
        assert!(score < 500); // Poor match
    }

    #[test]
    fn test_mastery_level_from_permille() {
        assert_eq!(MasteryLevel::from_permille(100), MasteryLevel::Novice);
        assert_eq!(MasteryLevel::from_permille(300), MasteryLevel::Beginner);
        assert_eq!(MasteryLevel::from_permille(500), MasteryLevel::Intermediate);
        assert_eq!(MasteryLevel::from_permille(700), MasteryLevel::Proficient);
        assert_eq!(MasteryLevel::from_permille(850), MasteryLevel::Advanced);
        assert_eq!(MasteryLevel::from_permille(980), MasteryLevel::Expert);
    }

    // ============== Retention Prediction Tests ==============

    #[test]
    fn test_calculate_stability_increases_with_reviews() {
        // More successful reviews = higher stability
        let stability_0 = calculate_stability(500, 0, 500, 0);
        let stability_1 = calculate_stability(500, 1, 500, 0);
        let stability_3 = calculate_stability(500, 3, 500, 0);

        assert!(stability_1 > stability_0, "1 review should increase stability");
        assert!(stability_3 > stability_1, "3 reviews should increase stability further");
    }

    #[test]
    fn test_calculate_stability_decreases_with_difficulty() {
        // Harder items have lower stability
        let easy = calculate_stability(500, 2, 200, 0);   // Low difficulty
        let medium = calculate_stability(500, 2, 500, 0); // Medium difficulty
        let hard = calculate_stability(500, 2, 800, 0);   // High difficulty

        assert!(easy > medium, "Easy items should have higher stability");
        assert!(medium > hard, "Hard items should have lower stability");
    }

    #[test]
    fn test_predict_retention_at_time() {
        let stability = 1440.0; // 1 day stability

        // At t=0, retention should be 1.0 (100%)
        let ret_0 = predict_retention_at_time(stability, 0.0);
        assert!((ret_0 - 1.0).abs() < 0.01, "Retention at t=0 should be ~100%");

        // At t=stability (1 day), retention should be ~37% (e^-1)
        let ret_1day = predict_retention_at_time(stability, 1440.0);
        assert!((ret_1day - 0.368).abs() < 0.05, "Retention at t=S should be ~37%");

        // Retention decreases over time
        let ret_1h = predict_retention_at_time(stability, 60.0);
        let ret_1d = predict_retention_at_time(stability, 1440.0);
        assert!(ret_1h > ret_1d, "Retention should decrease over time");
    }

    #[test]
    fn test_time_to_retention() {
        let stability = 1440.0; // 1 day stability

        // Time to 50% retention
        let t_50 = time_to_retention(stability, 0.5);
        assert!(t_50 > 0.0, "Time to 50% should be positive");
        assert!(t_50 < stability * 2.0, "Time to 50% should be reasonable");

        // Time to 80% should be less than time to 50%
        let t_80 = time_to_retention(stability, 0.8);
        assert!(t_80 < t_50, "Time to 80% should be less than time to 50%");
    }

    #[test]
    fn test_calculate_retrievability() {
        // High retention + high mastery = high retrievability
        let high = calculate_retrievability(0.9, 800, 700);
        let medium = calculate_retrievability(0.6, 500, 500);
        let low = calculate_retrievability(0.3, 200, 300);

        assert!(high > medium, "High retention should give high retrievability");
        assert!(medium > low, "Medium retention should beat low");
        assert!(high <= 1000, "Retrievability should be capped at 1000");
    }

    #[test]
    fn test_predict_retention_comprehensive() {
        // New learner, first review
        let new = predict_retention(300, 0, 500, 0, 0, 300);
        assert!(new.current_retention_permille > 900, "Just reviewed should be high");

        // Experienced learner with many reviews
        let experienced = predict_retention(800, 5, 300, 60, 1440, 800);
        assert!(experienced.stability_minutes > new.stability_minutes,
            "Experienced learner should have higher stability");

        // Overdue item
        let overdue = predict_retention(500, 2, 500, 10080, 1440, 500); // 1 week overdue
        assert!(overdue.current_retention_permille < 500,
            "Overdue item should have low retention");
    }

    #[test]
    fn test_optimal_review_schedule() {
        let schedule = optimal_review_schedule(1440.0, 800, 30); // 1 day stability, 80% target, 30 days

        assert!(!schedule.is_empty(), "Schedule should have reviews");
        assert!(schedule.len() < 30, "Shouldn't need 30 reviews in 30 days");

        // Reviews should be at increasing intervals
        for i in 1..schedule.len() {
            let interval_prev = if i == 1 { schedule[0] } else { schedule[i - 1] - schedule[i - 2] };
            let interval_curr = schedule[i] - schedule[i - 1];
            // Intervals grow because stability increases after each review
            assert!(interval_curr >= interval_prev / 2,
                "Intervals should grow or stay similar");
        }
    }

    #[test]
    fn test_retention_prediction_batch() {
        let skills = vec![
            (500, 2, 500, 60, 1440, 600),   // Medium skill, 1 hour ago
            (800, 5, 300, 120, 2880, 800),  // Strong skill, 2 hours ago
            (200, 0, 700, 4320, 0, 300),    // Weak skill, 3 days ago
        ];

        let predictions = predict_retention_batch(skills);
        assert_eq!(predictions.len(), 3);

        // Strong skill should have highest retention
        assert!(predictions[1].current_retention_permille > predictions[0].current_retention_permille);
        // Overdue weak skill should have lowest retention
        assert!(predictions[2].current_retention_permille < predictions[0].current_retention_permille);
    }
}
