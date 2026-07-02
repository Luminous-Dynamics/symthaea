# Mycelix Praxis: Differentiation Zomes API Reference

**Version**: 0.2.0
**Holochain**: 0.6.x
**Last Updated**: 2025-12-30

---

## Overview

The Praxis differentiation layer provides personalized, adaptive learning through four interconnected Holochain zomes:

| Zome | Purpose | Entry Types | Functions |
|------|---------|-------------|-----------|
| **SRS** | Spaced repetition (SM-2) | 6 | 18 |
| **Gamification** | XP, badges, streaks | 10 | 16 |
| **Adaptive** | BKT, ZPD, VARK intelligence | 10 | 22 |
| **Integration** | Cross-zome orchestration | 5 | 14 |

---

## Table of Contents

1. [SRS Zome API](#1-srs-zome-api)
2. [Gamification Zome API](#2-gamification-zome-api)
3. [Adaptive Learning Zome API](#3-adaptive-learning-zome-api)
4. [Integration Zome API](#4-integration-zome-api)
5. [Common Types & Patterns](#5-common-types--patterns)
6. [TypeScript Client Types](#6-typescript-client-types)
7. [Error Handling](#7-error-handling)

---

## 1. SRS Zome API

### Overview

Implements the **SuperMemo SM-2** algorithm for optimized memory retention through spaced repetition.

### Entry Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `ReviewCard` | Flashcard with scheduling metadata | `ease_factor_permille`, `interval_minutes`, `status` |
| `ReviewEvent` | Individual review record | `quality`, `time_taken_seconds`, `card_hash` |
| `ReviewSession` | Grouped review session | `cards_reviewed`, `accuracy_permille` |
| `DailyStats` | Aggregated daily statistics | `cards_reviewed`, `average_recall` |
| `SrsConfig` | SM-2 configuration parameters | `learning_steps_minutes`, `max_interval_days` |
| `Deck` | Card collection | `name`, `is_public`, `card_count` |

### Functions

#### Card Management

##### `create_card`
Create a new review card for a knowledge node.

```rust
#[hdk_extern]
pub fn create_card(input: CreateCardInput) -> ExternResult<Record>

pub struct CreateCardInput {
    pub node_hash: ActionHash,        // Knowledge node reference
    pub deck_hash: Option<ActionHash>, // Optional deck to add to
    pub custom_front: Option<String>,  // Custom question text
    pub custom_back: Option<String>,   // Custom answer text
    pub tags: Vec<String>,            // Organizational tags
}
```

**Returns**: Created `ReviewCard` record

---

##### `get_card`
Retrieve a card by its action hash.

```rust
#[hdk_extern]
pub fn get_card(action_hash: ActionHash) -> ExternResult<Option<Record>>
```

---

##### `get_my_cards`
Get all cards for the current learner.

```rust
#[hdk_extern]
pub fn get_my_cards(_: ()) -> ExternResult<Vec<Record>>
```

---

##### `get_due_cards`
Get cards due for review, sorted by priority.

```rust
#[hdk_extern]
pub fn get_due_cards(limit: u32) -> ExternResult<Vec<Record>>
```

**Parameters**:
- `limit`: Maximum number of cards to return

---

##### `suspend_card` / `unsuspend_card`
Temporarily pause or resume a card from reviews.

```rust
#[hdk_extern]
pub fn suspend_card(action_hash: ActionHash) -> ExternResult<Record>

#[hdk_extern]
pub fn unsuspend_card(action_hash: ActionHash) -> ExternResult<Record>
```

---

##### `bury_card`
Hide a card until the next day.

```rust
#[hdk_extern]
pub fn bury_card(action_hash: ActionHash) -> ExternResult<Record>
```

---

#### Review Processing

##### `submit_review`
Submit a review result and update card scheduling via SM-2.

```rust
#[hdk_extern]
pub fn submit_review(input: SubmitReviewInput) -> ExternResult<Record>

pub struct SubmitReviewInput {
    pub card_hash: ActionHash,
    pub quality: u8,              // 0-5 recall quality
    pub time_taken_seconds: u32,  // Time spent reviewing
    pub session_hash: Option<ActionHash>,
}
```

**Quality Scale**:
| Value | Meaning | Effect |
|-------|---------|--------|
| 0 | Complete blackout | Reset to learning |
| 1 | Incorrect, but recognized | Reset to learning |
| 2 | Incorrect, but easy to recall | Reset to learning |
| 3 | Correct with difficulty | Increase interval slowly |
| 4 | Correct with hesitation | Normal interval increase |
| 5 | Perfect recall | Maximum interval increase |

---

##### `get_card_reviews`
Get review history for a specific card.

```rust
#[hdk_extern]
pub fn get_card_reviews(card_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

#### Session Management

##### `start_session`
Begin a new review session.

```rust
#[hdk_extern]
pub fn start_session(pod_hash: Option<ActionHash>) -> ExternResult<Record>
```

---

##### `end_session`
Complete a review session with statistics.

```rust
#[hdk_extern]
pub fn end_session(input: EndSessionInput) -> ExternResult<Record>

pub struct EndSessionInput {
    pub session_hash: ActionHash,
    pub cards_reviewed: u32,
    pub correct_count: u32,
}
```

---

##### `get_my_sessions`
Get recent review sessions.

```rust
#[hdk_extern]
pub fn get_my_sessions(limit: u32) -> ExternResult<Vec<Record>>
```

---

#### Statistics & Forecasting

##### `get_stats`
Get daily statistics for a date range.

```rust
#[hdk_extern]
pub fn get_stats(input: GetStatsInput) -> ExternResult<Vec<Record>>

pub struct GetStatsInput {
    pub start_date: u32,  // YYYYMMDD format
    pub end_date: u32,
}
```

---

##### `get_forecast`
Predict future review workload.

```rust
#[hdk_extern]
pub fn get_forecast(days: u32) -> ExternResult<Vec<ReviewForecast>>

pub struct ReviewForecast {
    pub date: u32,           // YYYYMMDD
    pub cards_due: u32,
    pub estimated_minutes: u32,
}
```

---

#### Configuration

##### `get_or_create_config`
Get learner's SRS configuration or create default.

```rust
#[hdk_extern]
pub fn get_or_create_config(_: ()) -> ExternResult<Record>
```

---

##### `update_config`
Update SRS algorithm parameters.

```rust
#[hdk_extern]
pub fn update_config(input: UpdateConfigInput) -> ExternResult<Record>

pub struct UpdateConfigInput {
    pub config_hash: ActionHash,
    pub learning_steps_minutes: Option<Vec<u32>>,  // [1, 10] default
    pub graduating_interval_days: Option<u32>,      // 1 day default
    pub max_interval_days: Option<u32>,             // 365 default
    pub retention_target_permille: Option<u16>,     // 900 = 90%
    pub interval_modifier_permille: Option<u16>,    // 1000 = 100%
}
```

---

#### Deck Management

##### `create_deck`
Create a new card deck.

```rust
#[hdk_extern]
pub fn create_deck(input: CreateDeckInput) -> ExternResult<Record>

pub struct CreateDeckInput {
    pub name: String,
    pub description: String,
    pub is_public: bool,
    pub tags: Vec<String>,
}
```

---

##### `get_my_decks` / `get_public_decks`
List decks.

```rust
#[hdk_extern]
pub fn get_my_decks(_: ()) -> ExternResult<Vec<Record>>

#[hdk_extern]
pub fn get_public_decks(_: ()) -> ExternResult<Vec<Record>>
```

---

##### `get_deck_cards`
Get all cards in a deck.

```rust
#[hdk_extern]
pub fn get_deck_cards(deck_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

##### `add_card_to_deck`
Add an existing card to a deck.

```rust
#[hdk_extern]
pub fn add_card_to_deck(input: AddCardToDeckInput) -> ExternResult<()>

pub struct AddCardToDeckInput {
    pub card_hash: ActionHash,
    pub deck_hash: ActionHash,
}
```

---

## 2. Gamification Zome API

### Overview

Engagement system with **XP**, **badges**, **streaks**, and **leaderboards**.

### Entry Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `LearnerXp` | XP totals and level | `total_xp`, `level`, `daily_xp` |
| `XpTransaction` | Individual XP event | `amount`, `activity_type` |
| `LearnerStreak` | Daily login streak | `current_streak`, `longest_streak` |
| `DailyActivity` | Daily activity log | `date`, `xp_earned`, `active` |
| `BadgeDefinition` | Badge template | `name`, `rarity`, `requirements` |
| `EarnedBadge` | Earned badge instance | `badge_hash`, `earned_at` |
| `BadgeProgress` | Progress toward badges | `progress_permille` |
| `Leaderboard` | Competitive rankings | `type`, `period`, `scope` |
| `Reward` | Claimable rewards | `cost_xp`, `is_active` |
| `ClaimedReward` | Claimed reward record | `reward_hash`, `claimed_at` |

### Functions

#### XP Management

##### `get_or_create_xp`
Get or initialize learner's XP record.

```rust
#[hdk_extern]
pub fn get_or_create_xp(_: ()) -> ExternResult<Record>
```

---

##### `award_xp`
Award XP for an activity (automatically applies streak bonus).

```rust
#[hdk_extern]
pub fn award_xp(input: AwardXpInput) -> ExternResult<Record>

pub struct AwardXpInput {
    pub base_xp: u32,
    pub activity_type: XpActivityType,
    pub reference_hash: Option<ActionHash>,
    pub description: String,
}

pub enum XpActivityType {
    CardReview,
    QuizCompletion,
    LessonCompletion,
    ExerciseCompletion,
    ProjectSubmission,
    DailyLogin,
    StreakBonus,
    AchievementUnlock,
    PeerHelp,
    ContentCreation,
    Custom(String),
}
```

**XP Formula**: `final_xp = base_xp × streak_bonus / 1000`

---

##### `get_xp_transactions`
Get XP transaction history.

```rust
#[hdk_extern]
pub fn get_xp_transactions(limit: u32) -> ExternResult<Vec<Record>>
```

---

#### Streak Management

##### `get_or_create_streak`
Get or initialize streak record.

```rust
#[hdk_extern]
pub fn get_or_create_streak(_: ()) -> ExternResult<Record>
```

---

##### `freeze_streak`
Use a freeze to protect streak (limited uses).

```rust
#[hdk_extern]
pub fn freeze_streak(_: ()) -> ExternResult<Record>
```

---

#### Badge System

##### `create_badge_definition`
Create a new badge type (admin function).

```rust
#[hdk_extern]
pub fn create_badge_definition(input: CreateBadgeInput) -> ExternResult<Record>

pub struct CreateBadgeInput {
    pub name: String,
    pub description: String,
    pub icon_url: String,
    pub rarity: BadgeRarity,
    pub category: BadgeCategory,
    pub requirements: Vec<BadgeRequirement>,
    pub xp_reward: u32,
}

pub enum BadgeRarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
    Mythic,
    Unique,
}

pub enum BadgeCategory {
    Progress,
    Streak,
    Social,
    Challenge,
    Special,
    Mastery,
    Collection,
}
```

---

##### `award_badge`
Award a badge to a learner.

```rust
#[hdk_extern]
pub fn award_badge(input: AwardBadgeInput) -> ExternResult<Record>

pub struct AwardBadgeInput {
    pub badge_hash: ActionHash,
    pub reason: String,
}
```

---

##### `get_my_badges`
Get learner's earned badges.

```rust
#[hdk_extern]
pub fn get_my_badges(_: ()) -> ExternResult<Vec<Record>>
```

---

##### `get_all_badge_definitions`
Get all available badges.

```rust
#[hdk_extern]
pub fn get_all_badge_definitions(_: ()) -> ExternResult<Vec<Record>>
```

---

#### Leaderboards

##### `create_leaderboard`
Create a new leaderboard.

```rust
#[hdk_extern]
pub fn create_leaderboard(input: CreateLeaderboardInput) -> ExternResult<Record>

pub struct CreateLeaderboardInput {
    pub leaderboard_type: LeaderboardType,
    pub period: LeaderboardPeriod,
    pub scope: LeaderboardScope,
    pub name: String,
}

pub enum LeaderboardType { Xp, Streak, Badges, Reviews, Mastery }
pub enum LeaderboardPeriod { Daily, Weekly, Monthly, AllTime }
pub enum LeaderboardScope { Global, Course(ActionHash), Pod(ActionHash) }
```

---

##### `get_all_leaderboards` / `get_leaderboard_entries`
List leaderboards and entries.

```rust
#[hdk_extern]
pub fn get_all_leaderboards(_: ()) -> ExternResult<Vec<Record>>

#[hdk_extern]
pub fn get_leaderboard_entries(leaderboard_hash: ActionHash) -> ExternResult<Vec<Record>>
```

---

#### Rewards

##### `create_reward`
Create a claimable reward.

```rust
#[hdk_extern]
pub fn create_reward(input: CreateRewardInput) -> ExternResult<Record>

pub struct CreateRewardInput {
    pub name: String,
    pub description: String,
    pub cost_xp: u64,
    pub available_quantity: Option<u32>,
    pub expires_at: Option<i64>,
}
```

---

##### `claim_reward`
Claim a reward (spends XP).

```rust
#[hdk_extern]
pub fn claim_reward(reward_hash: ActionHash) -> ExternResult<Record>
```

---

##### `get_my_rewards` / `get_all_rewards`
List rewards.

```rust
#[hdk_extern]
pub fn get_my_rewards(_: ()) -> ExternResult<Vec<Record>>

#[hdk_extern]
pub fn get_all_rewards(_: ()) -> ExternResult<Vec<Record>>
```

---

#### Summary

##### `get_gamification_summary`
Get complete gamification status.

```rust
#[hdk_extern]
pub fn get_gamification_summary(_: ()) -> ExternResult<GamificationSummary>

pub struct GamificationSummary {
    pub total_xp: u64,
    pub level: u32,
    pub xp_to_next_level: u64,
    pub level_progress_permille: u16,
    pub current_streak: u32,
    pub longest_streak: u32,
    pub streak_bonus_permille: u16,
    pub badges_earned: u32,
    pub freezes_remaining: u8,
}
```

---

## 3. Adaptive Learning Zome API

### Overview

ML-inspired personalization using **Bayesian Knowledge Tracing (BKT)**, **Zone of Proximal Development (ZPD)**, and **VARK learning styles**.

### Entry Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `LearnerProfile` | Learning preferences | `preferred_style`, `cognitive_load_tolerance` |
| `LearningStyleAssessment` | VARK assessment | `visual_score_permille`, `auditory_score_permille`, ... |
| `SkillMastery` | Per-skill BKT mastery | `mastery_permille`, `p_learn`, `p_guess`, `p_slip` |
| `DifficultyCalibration` | Content difficulty | `difficulty_permille`, `attempts` |
| `Recommendation` | Content suggestions | `content_hash`, `reason`, `confidence_permille` |
| `LearningGoal` | Goal tracking | `target_hashes`, `priority`, `deadline` |
| `SessionAnalytics` | Per-session analysis | `focus_score_permille`, `flow_state_permille` |
| `AggregatedAnalytics` | Long-term trends | `avg_daily_minutes`, `mastery_velocity` |
| `AdaptivePath` | Learning journey | `steps`, `current_step_index` |
| `PathStep` | Path milestones | `content_hash`, `required_mastery_permille` |

### Functions

#### Profile Management

##### `create_profile`
Create learner profile.

```rust
#[hdk_extern]
pub fn create_profile(input: CreateProfileInput) -> ExternResult<ActionHash>

pub struct CreateProfileInput {
    pub display_name: String,
    pub preferred_style: LearningStyle,
    pub cognitive_load_tolerance: u16,  // 0-1000
    pub daily_goal_minutes: u32,
    pub timezone_offset_minutes: i32,
}

pub enum LearningStyle {
    Visual,
    Auditory,
    ReadingWriting,
    Kinesthetic,
    Multimodal,
}
```

---

##### `get_my_profile`
Get current learner's profile.

```rust
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<LearnerProfile>>
```

---

#### Learning Style Assessment

##### `record_style_assessment`
Record a VARK assessment result.

```rust
#[hdk_extern]
pub fn record_style_assessment(input: LearningStyleAssessment) -> ExternResult<ActionHash>

pub struct LearningStyleAssessment {
    pub visual_score_permille: u16,
    pub auditory_score_permille: u16,
    pub reading_writing_score_permille: u16,
    pub kinesthetic_score_permille: u16,
    pub completed_at: i64,
    pub assessment_version: String,
}
```

---

##### `calculate_learning_style`
Calculate dominant learning style from assessments.

```rust
#[hdk_extern]
pub fn calculate_learning_style(_: ()) -> ExternResult<LearningStyleResult>

pub struct LearningStyleResult {
    pub dominant_style: LearningStyle,
    pub scores: (u16, u16, u16, u16),  // (V, A, R, K)
    pub is_multimodal: bool,
    pub confidence_permille: u16,
}
```

---

#### Mastery Tracking (BKT)

##### `get_or_create_mastery`
Initialize mastery tracking for a skill.

```rust
#[hdk_extern]
pub fn get_or_create_mastery(input: CreateMasteryInput) -> ExternResult<ActionHash>

pub struct CreateMasteryInput {
    pub skill_hash: ActionHash,
    pub skill_name: String,
    pub initial_mastery_permille: Option<u16>,
}
```

---

##### `record_attempt`
Record a learning attempt and update BKT.

```rust
#[hdk_extern]
pub fn record_attempt(input: RecordAttemptInput) -> ExternResult<MasteryUpdateResult>

pub struct RecordAttemptInput {
    pub skill_hash: ActionHash,
    pub is_correct: bool,
    pub response_time_ms: u32,
    pub content_hash: Option<ActionHash>,
}

pub struct MasteryUpdateResult {
    pub skill_hash: ActionHash,
    pub old_mastery_permille: u16,
    pub new_mastery_permille: u16,
    pub level: MasteryLevel,
    pub level_changed: bool,
}
```

**BKT Update Formula**:
```
// Correct answer:
new_mastery = p_learn + (1.0 - p_learn) × p_transition

// Incorrect answer:
new_mastery = (p_learn × p_slip) / (p_learn × p_slip + (1.0 - p_learn) × (1.0 - p_guess))
```

---

##### `get_my_masteries`
Get all mastery records.

```rust
#[hdk_extern]
pub fn get_my_masteries(_: ()) -> ExternResult<Vec<SkillMastery>>
```

---

##### `get_due_for_review`
Get skills due for review (mastery decay).

```rust
#[hdk_extern]
pub fn get_due_for_review(_: ()) -> ExternResult<Vec<SkillMastery>>
```

---

#### Recommendations

##### `generate_recommendations`
Generate basic content recommendations.

```rust
#[hdk_extern]
pub fn generate_recommendations(input: GenerateRecsInput) -> ExternResult<Vec<Recommendation>>

pub struct GenerateRecsInput {
    pub content_pool: Vec<ContentCandidate>,
    pub limit: u32,
}

pub struct ContentCandidate {
    pub content_hash: ActionHash,
    pub content_type: ContentType,
    pub skill_hash: ActionHash,
    pub difficulty_permille: u16,
    pub estimated_minutes: u32,
    pub topic_hash: Option<ActionHash>,
}
```

---

##### `generate_smart_recommendations_v2`
Generate advanced recommendations with flow state, circadian rhythm, and VARK optimization.

```rust
#[hdk_extern]
pub fn generate_smart_recommendations_v2(input: SmartRecsInput) -> ExternResult<Vec<SmartRecommendation>>

pub struct SmartRecsInput {
    pub content_pool: Vec<ContentCandidate>,
    pub limit: u32,
    pub hour: Option<u8>,           // Current hour (0-23), defaults to system time
    pub day_of_week: Option<u8>,    // Day (0=Sun, 6=Sat)
    pub recent_topics: Option<Vec<ActionHash>>,  // For interleaving
    pub sessions_today: Option<u32>, // For energy estimation
}

pub struct SmartRecommendation {
    pub content_hash: ActionHash,
    pub content_type: ContentType,
    pub skill_hash: ActionHash,
    pub difficulty_permille: u16,
    pub smart_score_permille: u16,  // Combined optimization score
    pub reasons: Vec<RecommendationReason>,
    pub components: SmartScoreComponents,
}

pub struct SmartScoreComponents {
    pub flow_score: u16,         // ZPD alignment (25% weight)
    pub style_score: u16,        // VARK match (20% weight)
    pub timing_score: u16,       // Circadian bonus (15% weight)
    pub goal_score: u16,         // Goal alignment (25% weight)
    pub interleaving_score: u16, // Topic switching (15% weight)
}
```

**Smart Score Formula**:
```
smart_score = (flow×25 + style×20 + timing×15 + goal×25 + interleave×15) / 100
```

---

#### Goal Management

##### `create_goal`
Create a learning goal.

```rust
#[hdk_extern]
pub fn create_goal(input: CreateGoalInput) -> ExternResult<ActionHash>

pub struct CreateGoalInput {
    pub title: String,
    pub description: String,
    pub goal_type: GoalType,
    pub target_hashes: Vec<ActionHash>,
    pub target_value_permille: u16,
    pub deadline: Option<i64>,
    pub priority: GoalPriority,
}

pub enum GoalType { MasterSkill, CompleteContent, StudyTime, ReviewCards, Custom(String) }
pub enum GoalPriority { Critical, High, Medium, Low }
```

---

##### `get_my_goals`
Get active goals.

```rust
#[hdk_extern]
pub fn get_my_goals(_: ()) -> ExternResult<Vec<LearningGoal>>
```

---

##### `update_goal_progress`
Update goal progress.

```rust
#[hdk_extern]
pub fn update_goal_progress(input: UpdateGoalProgressInput) -> ExternResult<LearningGoal>

pub struct UpdateGoalProgressInput {
    pub goal_hash: ActionHash,
    pub progress_permille: u16,
}
```

---

#### Session Analytics

##### `record_session`
Record a learning session with analytics.

```rust
#[hdk_extern]
pub fn record_session(input: RecordSessionInput) -> ExternResult<ActionHash>

pub struct RecordSessionInput {
    pub started_at: i64,
    pub ended_at: i64,
    pub content_hashes: Vec<ActionHash>,
    pub focus_score_permille: u16,
    pub flow_state_permille: u16,
    pub interruptions: u32,
}
```

---

##### `get_recent_sessions`
Get recent session analytics.

```rust
#[hdk_extern]
pub fn get_recent_sessions(limit: u32) -> ExternResult<Vec<SessionAnalytics>>
```

---

#### Flow State Analysis

##### `analyze_flow_state`
Analyze current flow state based on learning patterns.

```rust
#[hdk_extern]
pub fn analyze_flow_state(input: AnalyzeFlowInput) -> ExternResult<FlowStateAnalysis>

pub struct AnalyzeFlowInput {
    pub skill_mastery_permille: u16,
    pub content_difficulty_permille: u16,
    pub session_duration_minutes: u32,
    pub recent_accuracy_permille: u16,
    pub recent_response_times_ms: Vec<u32>,
}

pub struct FlowStateAnalysis {
    pub state: FlowState,
    pub confidence_permille: u16,
    pub metrics: FlowMetrics,
    pub adjustments: Vec<FlowAdjustment>,
    pub trend: FlowTrend,
}

pub enum FlowState {
    Boredom,     // Skill >> Challenge
    Relaxation,  // Skill > Challenge
    Flow,        // Balanced (±100 permille)
    Arousal,     // Challenge > Skill
    Anxiety,     // Challenge >> Skill
    Overwhelm,   // Challenge >>> Skill
    Warming,     // Not enough data
    Fatigue,     // Long session, declining performance
}

pub enum FlowAdjustment {
    IncreaseDifficulty(u16),   // Permille to increase
    DecreaseDifficulty(u16),   // Permille to decrease
    TakeBreak(u32),            // Suggested break minutes
    SwitchTopic,               // Change topic for variety
    ShortenSession,            // Reduce session length
    ReduceChallengeScope,      // Simpler content
    IntroduceNewMaterial,      // More stimulating content
    ReviewBasics,              // Foundation reinforcement
}

pub enum FlowTrend {
    Improving,   // Moving toward flow
    Stable,      // Maintaining current state
    Declining,   // Moving away from flow
    Unknown,     // Insufficient data
}
```

---

##### `get_optimal_learning_window`
Get optimal session parameters based on history.

```rust
#[hdk_extern]
pub fn get_optimal_learning_window(_: ()) -> ExternResult<OptimalLearningWindow>

pub struct OptimalLearningWindow {
    pub recommended_duration_minutes: u32,
    pub optimal_difficulty_range: (u16, u16),  // (min, max) permille
    pub best_hours: Vec<u8>,                   // Best hours for learning
    pub suggested_content_types: Vec<ContentType>,
    pub estimated_energy_permille: u16,
    pub confidence_permille: u16,
}
```

---

#### Caching & Batch Operations

##### `get_learner_context`
Get combined learner data in a single call (optimized for UI).

```rust
#[hdk_extern]
pub fn get_learner_context(_: ()) -> ExternResult<LearnerContext>

pub struct LearnerContext {
    pub profile: Option<LearnerProfile>,
    pub masteries: Vec<SkillMastery>,
    pub goals: Vec<LearningGoal>,
    pub due_for_review: Vec<SkillMastery>,
    pub strengths: Vec<SkillMastery>,      // Top 5 by mastery
    pub weaknesses: Vec<SkillMastery>,     // Bottom 5 by mastery
    pub stats: LearnerStats,
}

pub struct LearnerStats {
    pub total_skills: u32,
    pub mastered_count: u32,              // >= 800 permille
    pub in_progress_count: u32,           // 200-799 permille
    pub novice_count: u32,                // < 200 permille
    pub avg_mastery_permille: u16,
    pub total_attempts: u32,
    pub overall_accuracy_permille: u16,
    pub dominant_style: LearningStyle,
}
```

---

##### `get_masteries_paginated`
Get masteries with pagination (for large datasets).

```rust
#[hdk_extern]
pub fn get_masteries_paginated(input: PaginatedMasteriesInput) -> ExternResult<PaginatedMasteriesResult>

pub struct PaginatedMasteriesInput {
    pub offset: u32,
    pub limit: u32,
    pub sort_by: MasterySortField,
    pub ascending: bool,
    pub filter_level: Option<MasteryLevel>,
}

pub enum MasterySortField { Mastery, LastAttempt, Attempts, Name }

pub struct PaginatedMasteriesResult {
    pub masteries: Vec<SkillMastery>,
    pub total_count: u32,
    pub has_more: bool,
}
```

---

#### Adaptive Paths

##### `create_adaptive_path`
Create a personalized learning path.

```rust
#[hdk_extern]
pub fn create_adaptive_path(input: CreatePathInput) -> ExternResult<ActionHash>

pub struct CreatePathInput {
    pub name: String,
    pub description: String,
    pub steps: Vec<PathStepInput>,
    pub is_adaptive: bool,  // Auto-adjust based on progress
}
```

---

##### `advance_path_step`
Advance to the next step in a path.

```rust
#[hdk_extern]
pub fn advance_path_step(path_hash: ActionHash) -> ExternResult<AdaptivePath>
```

---

##### `get_my_paths`
Get learner's adaptive paths.

```rust
#[hdk_extern]
pub fn get_my_paths(_: ()) -> ExternResult<Vec<AdaptivePath>>
```

---

#### Difficulty Calibration

##### `update_difficulty_calibration`
Update content difficulty based on learner performance.

```rust
#[hdk_extern]
pub fn update_difficulty_calibration(input: RecordCalibrationInput) -> ExternResult<DifficultyCalibration>

pub struct RecordCalibrationInput {
    pub content_hash: ActionHash,
    pub was_successful: bool,
    pub time_taken_ms: u32,
    pub learner_mastery_at_attempt: u16,
}
```

---

#### Summary

##### `get_learner_summary`
Get comprehensive learner summary.

```rust
#[hdk_extern]
pub fn get_learner_summary(_: ()) -> ExternResult<LearnerSummary>

pub struct LearnerSummary {
    pub profile: Option<LearnerProfile>,
    pub learning_style: LearningStyle,
    pub mastery_count: u32,
    pub avg_mastery_permille: u16,
    pub skills_mastered: u32,
    pub due_for_review: u32,
    pub active_goals: u32,
    pub total_sessions: u32,
    pub total_learning_minutes: u32,
    pub active_paths: u32,
    pub recommendations_count: u32,
}
```

---

#### Performance Metrics

##### `collect_performance_metrics`
Collect comprehensive performance metrics for algorithm accuracy and learning effectiveness.

```rust
#[hdk_extern]
pub fn collect_performance_metrics(input: CollectMetricsInput) -> ExternResult<PerformanceMetrics>

pub struct CollectMetricsInput {
    /// Period to analyze (hours, default 24)
    pub period_hours: Option<u32>,
}

pub struct PerformanceMetrics {
    pub collected_at: i64,
    pub period_hours: u32,
    pub algorithm_metrics: AlgorithmMetrics,
    pub query_metrics: QueryMetrics,
    pub effectiveness_metrics: EffectivenessMetrics,
}

pub struct AlgorithmMetrics {
    pub bkt_accuracy_permille: u16,            // BKT prediction accuracy
    pub bkt_predictions: u32,                  // Total predictions made
    pub flow_prediction_accuracy_permille: u16, // Flow state accuracy
    pub recommendation_ctr_permille: u16,       // Recommendation click-through
    pub avg_followed_smart_score: u16,          // Score of followed recommendations
    pub avg_skipped_smart_score: u16,           // Score of skipped recommendations
}

pub struct QueryMetrics {
    pub avg_context_response_us: u32,          // get_learner_context latency
    pub avg_recommendation_response_us: u32,    // Recommendation latency
    pub total_calls: u32,                       // Total zome calls
    pub mastery_cache_hit_permille: u16,        // Cache effectiveness
    pub cross_zome_calls: u32,                  // Inter-zome calls
    pub peak_concurrent_ops: u32,               // Max concurrent operations
}

pub struct EffectivenessMetrics {
    pub avg_mastery_gain_per_session: u16,      // Learning rate
    pub avg_time_to_proficiency_minutes: u32,   // Time to 600 mastery
    pub retention_rate_permille: u16,           // Skills above threshold
    pub goal_completion_rate_permille: u16,     // Goals achieved
    pub avg_session_flow_score: u16,            // Average flow balance
    pub streak_maintenance_rate_permille: u16,  // Streak retention
}
```

---

##### `run_benchmarks`
Run performance benchmarks for key operations (development/testing).

```rust
#[hdk_extern]
pub fn run_benchmarks(_: ()) -> ExternResult<BenchmarkSummary>

pub struct BenchmarkSummary {
    pub run_at: i64,
    pub total_duration_us: u64,
    pub benchmarks: Vec<BenchmarkResult>,
    pub recommendations: Vec<String>,  // Performance improvement suggestions
}

pub struct BenchmarkResult {
    pub operation: String,   // Operation name
    pub samples: u32,        // Number of samples
    pub min_us: u64,         // Minimum latency
    pub max_us: u64,         // Maximum latency
    pub avg_us: u64,         // Average latency
    pub p50_us: u64,         // 50th percentile
    pub p95_us: u64,         // 95th percentile
    pub p99_us: u64,         // 99th percentile
}
```

**Example Benchmark Output**:
```json
{
  "run_at": 1735574400000000,
  "total_duration_us": 2500000,
  "benchmarks": [
    { "operation": "get_my_masteries", "avg_us": 45000, "p95_us": 78000 },
    { "operation": "get_learner_context", "avg_us": 180000, "p95_us": 250000 },
    { "operation": "calculate_learning_style", "avg_us": 12000, "p95_us": 18000 }
  ],
  "recommendations": []
}
```

---

## 4. Integration Zome API

### Overview

Orchestrates cross-zome learning experiences, combining SRS, Gamification, and Adaptive systems.

### Entry Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `LearningEvent` | Unified event | `event_type`, `source_zome`, `xp_awarded` |
| `LearnerProgressAggregate` | Combined snapshot | `total_xp`, `mastery_avg`, `cards_due` |
| `AchievementTrigger` | Conditional badge triggers | `conditions`, `rewards` |
| `OrchestratedSession` | Coordinated session | `planned_activities`, `state` |
| `DailyLearningReport` | Daily summary | `events`, `achievements`, `insights` |

### Functions

#### Event Processing

##### `record_learning_event`
Record a unified learning event and process rewards.

```rust
#[hdk_extern]
pub fn record_learning_event(input: RecordEventInput) -> ExternResult<LearningEventResult>

pub struct RecordEventInput {
    pub event_type: LearningEventType,
    pub source_zome: SourceZome,
    pub reference_hash: ActionHash,
    pub metadata: serde_json::Value,
}

pub enum LearningEventType {
    CardReviewed,
    LessonCompleted,
    QuizCompleted,
    ExerciseCompleted,
    ProjectSubmitted,
    GoalAchieved,
    MasteryLevelUp,
    StreakMaintained,
    BadgeEarned,
    PathAdvanced,
    SessionCompleted,
    Custom(String),
}

pub enum SourceZome { Srs, Gamification, Adaptive, Learning, Fl, Credential, Dao, Integration }

pub struct LearningEventResult {
    pub event_hash: ActionHash,
    pub xp_awarded: u32,
    pub achievements_unlocked: Vec<String>,
    pub progress_update: Option<ProgressDelta>,
}
```

---

##### `get_recent_events`
Get recent learning events.

```rust
#[hdk_extern]
pub fn get_recent_events(limit: u32) -> ExternResult<Vec<LearningEvent>>
```

---

#### Progress Aggregation

##### `update_progress_aggregate`
Manually update progress aggregate.

```rust
#[hdk_extern]
pub fn update_progress_aggregate(input: UpdateProgressInput) -> ExternResult<ActionHash>

pub struct UpdateProgressInput {
    pub srs_stats: Option<SrsProgressStats>,
    pub gamification_stats: Option<GamificationProgressStats>,
    pub adaptive_stats: Option<AdaptiveProgressStats>,
}
```

---

##### `get_progress_aggregate`
Get current progress snapshot.

```rust
#[hdk_extern]
pub fn get_progress_aggregate(_: ()) -> ExternResult<Option<LearnerProgressAggregate>>
```

---

##### `refresh_progress_from_zomes`
Refresh aggregate by pulling data from all zomes.

```rust
#[hdk_extern]
pub fn refresh_progress_from_zomes(_: ()) -> ExternResult<CrossZomeProgressResult>

pub struct CrossZomeProgressResult {
    pub srs_cards_total: u32,
    pub srs_cards_due: u32,
    pub gamification: GamificationSummary,
    pub adaptive: AdaptiveLearnerSummary,
    pub last_refreshed: i64,
}
```

---

#### Unified Dashboard

##### `get_unified_dashboard`
Get all data for the main dashboard UI.

```rust
#[hdk_extern]
pub fn get_unified_dashboard(_: ()) -> ExternResult<UnifiedDashboard>

pub struct UnifiedDashboard {
    pub progress: Option<LearnerProgressAggregate>,
    pub gamification: GamificationSummary,
    pub adaptive: AdaptiveLearnerSummary,
    pub recent_events: Vec<LearningEvent>,
    pub active_session: Option<OrchestratedSession>,
    pub recommendations: Vec<DashboardRecommendation>,
}
```

---

#### Session Orchestration

##### `start_orchestrated_session`
Start a coordinated learning session with planned activities.

```rust
#[hdk_extern]
pub fn start_orchestrated_session(input: StartSessionInput) -> ExternResult<ActionHash>

pub struct StartSessionInput {
    pub planned_duration_minutes: u32,
    pub focus_areas: Vec<ActionHash>,      // Skill/topic hashes
    pub include_srs: bool,
    pub include_lessons: bool,
    pub include_exercises: bool,
}
```

---

##### `complete_session_activity`
Mark an activity as complete within a session.

```rust
#[hdk_extern]
pub fn complete_session_activity(input: CompleteActivityInput) -> ExternResult<OrchestratedSession>

pub struct CompleteActivityInput {
    pub session_hash: ActionHash,
    pub activity_index: u32,
    pub result: ActivityResult,
}
```

---

##### `end_orchestrated_session`
End a session and generate summary.

```rust
#[hdk_extern]
pub fn end_orchestrated_session(session_hash: ActionHash) -> ExternResult<OrchestratedSession>
```

---

##### `get_active_session`
Get currently active orchestrated session.

```rust
#[hdk_extern]
pub fn get_active_session(_: ()) -> ExternResult<Option<OrchestratedSession>>
```

---

#### Daily Reports

##### `generate_daily_report`
Generate a comprehensive daily report.

```rust
#[hdk_extern]
pub fn generate_daily_report(_: ()) -> ExternResult<ActionHash>
```

---

##### `get_recent_reports`
Get recent daily reports.

```rust
#[hdk_extern]
pub fn get_recent_reports(days: u32) -> ExternResult<Vec<DailyLearningReport>>
```

---

#### Achievement Triggers

##### `create_achievement_trigger`
Create a conditional achievement trigger.

```rust
#[hdk_extern]
pub fn create_achievement_trigger(trigger: AchievementTrigger) -> ExternResult<ActionHash>

pub struct AchievementTrigger {
    pub name: String,
    pub description: String,
    pub conditions: Vec<TriggerCondition>,
    pub rewards: Vec<TriggerReward>,
    pub is_active: bool,
    pub is_recurring: bool,
}

pub struct TriggerCondition {
    pub metric: TriggerMetric,
    pub comparison: ComparisonOp,
    pub threshold: u64,
}

pub enum TriggerMetric {
    TotalXp,
    CurrentStreak,
    CardsReviewed,
    SkillsMastered,
    GoalsCompleted,
    SessionsCompleted,
    TotalLearningMinutes,
    DailyLoginCount,
    Custom(String),
}

pub enum ComparisonOp { Eq, Gt, Gte, Lt, Lte, Ne }
```

---

##### `get_achievement_triggers`
Get all achievement triggers.

```rust
#[hdk_extern]
pub fn get_achievement_triggers(_: ()) -> ExternResult<Vec<AchievementTrigger>>
```

---

## 5. Common Types & Patterns

### Permille Convention

All percentages are represented as **permille** (0-1000) instead of floats for `Eq` compatibility:

```rust
// 75% accuracy
pub accuracy_permille: u16 = 750;

// 2.5 ease factor
pub ease_factor_permille: u16 = 2500;
```

### Timestamp Handling

Timestamps are stored as `i64` microseconds since Unix epoch:

```rust
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}
```

### Link Resolution Pattern

```rust
// Correct pattern for Holochain 0.6
if let Some(action_hash) = link.target.into_action_hash() {
    if let Some(record) = get(action_hash, GetOptions::default())? {
        // Process record
    }
}
```

### Path-Based Anchors

```rust
fn learner_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("zome.category.{}", agent));
    let typed_path = path.typed(LinkTypes::MyLinkType)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}
```

### Mastery Levels

```rust
pub enum MasteryLevel {
    Novice,       // 0-199
    Beginner,     // 200-399
    Competent,    // 400-599
    Proficient,   // 600-799
    Expert,       // 800-949
    Master,       // 950-1000
}
```

### Content Types

```rust
pub enum ContentType {
    Lesson,      // New concepts, reading
    Quiz,        // Quick knowledge checks
    Exercise,    // Hands-on practice
    Project,     // Extended application
    Assessment,  // Formal evaluation
    Challenge,   // Advanced problems
}
```

---

## 6. TypeScript Client Types

Full TypeScript types are available in `apps/web/src/types/zomes.ts` (534 lines).

### Usage Example

```typescript
import { ZomeFunctions, SrsZomeFunctions } from './types/zomes';

// Create a card
const card = await client.callZome({
  zome_name: 'srs_coordinator',
  fn_name: 'create_card',
  payload: {
    node_hash: knowledgeNodeHash,
    custom_front: 'What is photosynthesis?',
    custom_back: 'The process plants use to convert light to energy',
    tags: ['biology', 'plants'],
  } as CreateCardInput,
});

// Submit a review
const result = await client.callZome({
  zome_name: 'srs_coordinator',
  fn_name: 'submit_review',
  payload: {
    card_hash: cardHash,
    quality: 4, // Good recall
    time_taken_seconds: 5,
  } as SubmitReviewInput,
});
```

---

## 7. Error Handling

### Structured Error Codes

| Code | Category | Example |
|------|----------|---------|
| 1xx | Entity | `EntityNotFound (100)`, `EntityAlreadyExists (101)` |
| 2xx | Validation | `ValidationFailed (200)`, `OutOfRange (202)` |
| 3xx | Authorization | `Unauthorized (300)`, `AuthorMismatch (302)` |
| 4xx | State | `InvalidStateTransition (400)` |
| 5xx | External | `CrossZomeCallFailed (500)`, `NetworkError (501)` |
| 6xx | Resource | `LimitExceeded (601)`, `QuotaExceeded (602)` |

### Error Format

```
[E100] ReviewCard get failed: Entry not found in DHT (hash: Qm123abc) → Hint: The entry may have been deleted or never created
```

### Usage in Zomes

```rust
use praxis_core::errors::{srs_errors, PraxisError};

fn to_wasm_error(err: PraxisError) -> WasmError {
    wasm_error!(WasmErrorInner::Guest(err.to_message()))
}

let record = get(hash.clone(), GetOptions::default())?
    .ok_or_else(|| to_wasm_error(srs_errors::card_not_found(&hash.to_string())))?;
```

---

## Appendix: Quick Reference

### SRS Functions
| Function | Input | Output |
|----------|-------|--------|
| `create_card` | `CreateCardInput` | `Record` |
| `get_card` | `ActionHash` | `Option<Record>` |
| `get_my_cards` | `()` | `Vec<Record>` |
| `get_due_cards` | `u32` | `Vec<Record>` |
| `suspend_card` | `ActionHash` | `Record` |
| `unsuspend_card` | `ActionHash` | `Record` |
| `bury_card` | `ActionHash` | `Record` |
| `submit_review` | `SubmitReviewInput` | `Record` |
| `get_card_reviews` | `ActionHash` | `Vec<Record>` |
| `start_session` | `Option<ActionHash>` | `Record` |
| `end_session` | `EndSessionInput` | `Record` |
| `get_my_sessions` | `u32` | `Vec<Record>` |
| `get_stats` | `GetStatsInput` | `Vec<Record>` |
| `get_forecast` | `u32` | `Vec<ReviewForecast>` |
| `get_or_create_config` | `()` | `Record` |
| `update_config` | `UpdateConfigInput` | `Record` |
| `create_deck` | `CreateDeckInput` | `Record` |
| `get_my_decks` | `()` | `Vec<Record>` |

### Gamification Functions
| Function | Input | Output |
|----------|-------|--------|
| `get_or_create_xp` | `()` | `Record` |
| `award_xp` | `AwardXpInput` | `Record` |
| `get_xp_transactions` | `u32` | `Vec<Record>` |
| `get_or_create_streak` | `()` | `Record` |
| `freeze_streak` | `()` | `Record` |
| `create_badge_definition` | `CreateBadgeInput` | `Record` |
| `award_badge` | `AwardBadgeInput` | `Record` |
| `get_my_badges` | `()` | `Vec<Record>` |
| `get_all_badge_definitions` | `()` | `Vec<Record>` |
| `create_leaderboard` | `CreateLeaderboardInput` | `Record` |
| `get_all_leaderboards` | `()` | `Vec<Record>` |
| `get_leaderboard_entries` | `ActionHash` | `Vec<Record>` |
| `create_reward` | `CreateRewardInput` | `Record` |
| `claim_reward` | `ActionHash` | `Record` |
| `get_my_rewards` | `()` | `Vec<Record>` |
| `get_gamification_summary` | `()` | `GamificationSummary` |

### Adaptive Functions
| Function | Input | Output |
|----------|-------|--------|
| `create_profile` | `CreateProfileInput` | `ActionHash` |
| `get_my_profile` | `()` | `Option<LearnerProfile>` |
| `record_style_assessment` | `LearningStyleAssessment` | `ActionHash` |
| `calculate_learning_style` | `()` | `LearningStyleResult` |
| `get_or_create_mastery` | `CreateMasteryInput` | `ActionHash` |
| `record_attempt` | `RecordAttemptInput` | `MasteryUpdateResult` |
| `get_my_masteries` | `()` | `Vec<SkillMastery>` |
| `get_due_for_review` | `()` | `Vec<SkillMastery>` |
| `generate_recommendations` | `GenerateRecsInput` | `Vec<Recommendation>` |
| `generate_smart_recommendations_v2` | `SmartRecsInput` | `Vec<SmartRecommendation>` |
| `create_goal` | `CreateGoalInput` | `ActionHash` |
| `get_my_goals` | `()` | `Vec<LearningGoal>` |
| `update_goal_progress` | `UpdateGoalProgressInput` | `LearningGoal` |
| `record_session` | `RecordSessionInput` | `ActionHash` |
| `get_recent_sessions` | `u32` | `Vec<SessionAnalytics>` |
| `analyze_flow_state` | `AnalyzeFlowInput` | `FlowStateAnalysis` |
| `get_optimal_learning_window` | `()` | `OptimalLearningWindow` |
| `get_learner_context` | `()` | `LearnerContext` |
| `get_masteries_paginated` | `PaginatedMasteriesInput` | `PaginatedMasteriesResult` |
| `create_adaptive_path` | `CreatePathInput` | `ActionHash` |
| `advance_path_step` | `ActionHash` | `AdaptivePath` |
| `get_learner_summary` | `()` | `LearnerSummary` |
| `collect_performance_metrics` | `CollectMetricsInput` | `PerformanceMetrics` |
| `run_benchmarks` | `()` | `BenchmarkSummary` |

### Integration Functions
| Function | Input | Output |
|----------|-------|--------|
| `record_learning_event` | `RecordEventInput` | `LearningEventResult` |
| `get_recent_events` | `u32` | `Vec<LearningEvent>` |
| `update_progress_aggregate` | `UpdateProgressInput` | `ActionHash` |
| `get_progress_aggregate` | `()` | `Option<LearnerProgressAggregate>` |
| `refresh_progress_from_zomes` | `()` | `CrossZomeProgressResult` |
| `get_unified_dashboard` | `()` | `UnifiedDashboard` |
| `start_orchestrated_session` | `StartSessionInput` | `ActionHash` |
| `complete_session_activity` | `CompleteActivityInput` | `OrchestratedSession` |
| `end_orchestrated_session` | `ActionHash` | `OrchestratedSession` |
| `get_active_session` | `()` | `Option<OrchestratedSession>` |
| `generate_daily_report` | `()` | `ActionHash` |
| `get_recent_reports` | `u32` | `Vec<DailyLearningReport>` |
| `create_achievement_trigger` | `AchievementTrigger` | `ActionHash` |
| `get_achievement_triggers` | `()` | `Vec<AchievementTrigger>` |

---

*Built with the Sacred Trinity: Human Vision + Claude Code + Holochain*
