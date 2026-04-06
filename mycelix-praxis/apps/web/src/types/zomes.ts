// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TypeScript type definitions for Mycelix EduNet Holochain zome calls
 *
 * This file defines the types for all entry types and function calls across
 * the four core zomes: Learning, FL, Credential, and DAO.
 */

// =============================================================================
// Common Types
// =============================================================================

export type ActionHash = Uint8Array;
export type AgentPubKey = Uint8Array;
export type Timestamp = number; // Unix timestamp in seconds

// =============================================================================
// Learning Zome Types
// =============================================================================

export interface Course {
  course_id: string;
  title: string;
  description: string;
  creator: string;
  tags: string[];
  model_id: string | null;
  created_at: number;
  updated_at: number;
  metadata: Record<string, unknown> | null;
}

export interface LearnerProgress {
  course_id: string;
  learner: string;
  progress_percent: number;
  completed_items: string[];
  model_version: string | null;
  last_active: number;
  metadata: Record<string, unknown> | null;
}

export interface LearningActivity {
  course_id: string;
  activity_type: string;
  item_id: string;
  outcome: number | null;
  duration_secs: number;
  timestamp: number;
}

export interface LearningZomeFunctions {
  // Course management
  create_course: (course: Course) => Promise<ActionHash>;
  get_course: (course_hash: ActionHash) => Promise<Course | null>;
  list_courses: () => Promise<Course[]>;

  // Enrollment
  enroll: (course_action_hash: ActionHash) => Promise<void>;
  get_enrolled_courses: () => Promise<Course[]>;
  get_course_enrollments: (course_action_hash: ActionHash) => Promise<AgentPubKey[]>;

  // Learner progress tracking
  update_progress: (progress: LearnerProgress) => Promise<ActionHash>;
  get_progress: (action_hash: ActionHash) => Promise<LearnerProgress | null>;

  // Activity logging
  record_activity: (activity: LearningActivity) => Promise<ActionHash>;
}

// =============================================================================
// FL (Federated Learning) Zome Types
// =============================================================================

export interface FlRound {
  round_id: string;
  model_id: string;
  round_number: number;
  created_at: Timestamp;
  deadline: Timestamp;
  min_participants: number;
  current_participants: number;
  aggregation_method: string;
  privacy_params: PrivacyParams;
  status: RoundStatus;
  global_model_hash?: string;
}

export interface PrivacyParams {
  gradient_clip_norm: number;
  differential_privacy_epsilon?: number;
  differential_privacy_delta?: number;
  secure_aggregation: boolean;
}

export type RoundStatus = 'Active' | 'Aggregating' | 'Complete' | 'Failed';

export interface FlUpdate {
  update_id: string;
  round_id: string;
  participant_id: string;
  submitted_at: Timestamp;
  gradient_hash: string;
  num_samples: number;
  local_loss: number;
  commitment: string;
}

export interface FlZomeFunctions {
  // Round management
  create_fl_round: (round: FlRound) => Promise<ActionHash>;
  get_fl_round: (round_hash: ActionHash) => Promise<FlRound | null>;
  get_active_rounds: (model_id: string) => Promise<FlRound[]>;

  // Update submission
  submit_update: (update: FlUpdate) => Promise<ActionHash>;
  get_round_updates: (round_id: string) => Promise<FlUpdate[]>;

  // Aggregation
  aggregate_round: (round_id: string) => Promise<ActionHash>;
  get_aggregated_model: (round_id: string) => Promise<HolochainRecord | null>;
}

// Type alias for Holochain Record (matches @holochain/client)
export type HolochainRecord = {
  signed_action: {
    hashed: {
      content: unknown;
      hash: ActionHash;
    };
    signature: Uint8Array;
  };
  entry: {
    Present?: unknown;
  };
};

// =============================================================================
// Credential Zome Types
// =============================================================================

export interface VerifiableCredential {
  context: string; // W3C VC context URL
  credential_type: string[];
  issuer: string; // DID
  issuance_date: string; // ISO 8601
  expiration_date?: string; // ISO 8601

  // Credential Subject (flattened)
  subject_id: string; // DID
  course_id: string;
  model_id?: string;
  rubric_id?: string;
  score?: number;
  score_band: string;
  subject_metadata?: string; // JSON

  // Credential Status (flattened)
  status_id?: string;
  status_type?: string;
  status_list_index?: number;
  status_purpose?: string;

  // Proof (flattened)
  proof_type: string;
  proof_created: string; // ISO 8601
  verification_method: string;
  proof_purpose: string;
  proof_value: string; // Signature
}

export interface CredentialZomeFunctions {
  // Issuance
  issue_credential: (credential: VerifiableCredential) => Promise<ActionHash>;

  // Retrieval
  get_credential: (credential_hash: ActionHash) => Promise<VerifiableCredential | null>;
  get_learner_credentials: (learner_did: string) => Promise<VerifiableCredential[]>;
  get_course_credentials: (course_id: string) => Promise<VerifiableCredential[]>;
  get_issuer_credentials: (issuer_did: string) => Promise<VerifiableCredential[]>;

  // Verification
  verify_credential: (credential_hash: ActionHash) => Promise<boolean>;

  // Revocation
  revoke_credential: (credential_hash: ActionHash) => Promise<ActionHash>;
}

// =============================================================================
// DAO Zome Types
// =============================================================================

export type ProposalType = 'Fast' | 'Normal' | 'Slow';

export type ProposalCategory =
  | 'Curriculum'
  | 'Protocol'
  | 'Credentials'
  | 'Treasury'
  | 'Governance'
  | 'Emergency';

export type ProposalStatus =
  | 'Active'
  | 'Approved'
  | 'Executed'
  | 'Rejected'
  | 'Cancelled'
  | 'Vetoed';

export type VoteChoice = 'For' | 'Against' | 'Abstain';

export interface Proposal {
  proposal_id: string;
  title: string;
  description: string;
  proposer: string; // DID
  proposal_type: ProposalType;
  category: ProposalCategory;
  created_at: Timestamp;
  voting_deadline: Timestamp;
  actions_json: string; // JSON string of actions to execute
  votes_for: number;
  votes_against: number;
  votes_abstain: number;
  status: ProposalStatus;
}

export interface Vote {
  proposal_id: string;
  voter: string; // DID
  choice: VoteChoice;
  voting_power: number;
  timestamp: Timestamp;
  justification?: string;
}

export interface DaoZomeFunctions {
  // Proposal management
  create_proposal: (proposal: Proposal) => Promise<ActionHash>;
  get_proposal: (proposal_hash: ActionHash) => Promise<Proposal | null>;
  get_proposals_by_category: (category: ProposalCategory) => Promise<Proposal[]>;
  get_agent_proposals: (agent_did: string) => Promise<Proposal[]>;
  list_proposals: () => Promise<Proposal[]>;

  // Voting
  cast_vote: (vote: Vote) => Promise<ActionHash>;
  get_agent_votes: (agent_did: string) => Promise<Vote[]>;
}

// =============================================================================
// SRS (Spaced Repetition System) Zome Types
// =============================================================================

export type RecallQuality = 'Blackout' | 'Incorrect' | 'IncorrectEasy' | 'Hard' | 'Good' | 'Easy';

export type CardStatus = 'New' | 'Learning' | 'Review' | 'Relearning' | 'Suspended' | 'Buried';

export interface ReviewCard {
  deck_hash: ActionHash;
  front: string;
  back: string;
  tags: string[];
  status: CardStatus;
  ease_factor_permille: number; // 1000-5000 (1.0-5.0)
  interval_days: number;
  step_index: number;
  next_review_at: Timestamp;
  lapses: number;
  review_count: number;
  created_at: Timestamp;
  last_reviewed_at?: Timestamp;
}

export interface ReviewEvent {
  card_hash: ActionHash;
  quality: RecallQuality;
  reviewed_at: Timestamp;
  interval_before: number;
  interval_after: number;
  ease_before_permille: number;
  ease_after_permille: number;
  time_taken_ms: number;
}

export interface ReviewSession {
  deck_hash?: ActionHash;
  started_at: Timestamp;
  ended_at?: Timestamp;
  cards_reviewed: number;
  cards_new: number;
  cards_learning: number;
  accuracy_permille: number;
  avg_time_ms: number;
}

export interface Deck {
  name: string;
  description: string;
  settings: string; // JSON of SrsConfig
  created_at: Timestamp;
  card_count: number;
}

export interface SrsConfig {
  learning_steps_minutes: number[];
  graduating_interval_days: number;
  easy_interval_days: number;
  new_cards_per_day: number;
  review_cards_per_day: number;
  max_interval_days: number;
  starting_ease_permille: number;
  target_retention_permille: number;
  hard_interval_multiplier_permille: number;
  easy_bonus_permille: number;
  interval_modifier_permille: number;
  leech_threshold: number;
  leech_action: LeechAction;
}

export type LeechAction = 'Suspend' | 'Tag' | 'Notify' | 'Nothing';

export interface SrsZomeFunctions {
  create_deck: (input: { name: string; description: string }) => Promise<ActionHash>;
  get_deck: (deck_hash: ActionHash) => Promise<Deck | null>;
  get_my_decks: () => Promise<Deck[]>;
  create_card: (input: { deck_hash: ActionHash; front: string; back: string; tags: string[] }) => Promise<ActionHash>;
  get_card: (card_hash: ActionHash) => Promise<ReviewCard | null>;
  get_due_cards: (input: { deck_hash?: ActionHash; limit: number }) => Promise<ReviewCard[]>;
  review_card: (input: { card_hash: ActionHash; quality: RecallQuality }) => Promise<ReviewCard>;
  start_session: (deck_hash?: ActionHash) => Promise<ActionHash>;
  end_session: (session_hash: ActionHash) => Promise<ReviewSession>;
  get_stats: () => Promise<{ cards_total: number; cards_due: number; reviews_today: number; accuracy_permille: number }>;
}

// =============================================================================
// Gamification Zome Types
// =============================================================================

export type XpActivityType =
  | 'LessonComplete'
  | 'QuizPass'
  | 'QuizFail'
  | 'AssignmentSubmit'
  | 'PeerHelp'
  | 'Contribution'
  | 'StreakBonus'
  | 'AchievementUnlock'
  | 'DailyLogin'
  | 'ChallengComplete'
  | 'ProjectSubmit'
  | 'ReviewCard';

export type BadgeRarity = 'Common' | 'Uncommon' | 'Rare' | 'Epic' | 'Legendary' | 'Mythic' | 'Unique';

export type BadgeCategory = 'Learning' | 'Social' | 'Streak' | 'Mastery' | 'Challenge' | 'Community' | 'Special';

export interface LearnerXp {
  learner: AgentPubKey;
  total_xp: bigint;
  level: number;
  xp_to_next_level: number;
  lifetime_xp: bigint;
  multiplier_permille: number;
  multiplier_expires_at?: Timestamp;
  last_activity_at: Timestamp;
  created_at: Timestamp;
}

export interface XpTransaction {
  learner: AgentPubKey;
  amount: number;
  activity_type: XpActivityType;
  source_hash: ActionHash;
  multiplier_applied_permille: number;
  description: string;
  occurred_at: Timestamp;
}

export interface LearnerStreak {
  learner: AgentPubKey;
  current_streak: number;
  longest_streak: number;
  last_activity_date: number; // YYYYMMDD format
  streak_start_date: number;
  freeze_available: boolean;
  freeze_count_used: number;
  streak_broken_count: number;
}

export interface BadgeDefinition {
  badge_id: string;
  name: string;
  description: string;
  icon_url: string;
  rarity: BadgeRarity;
  category: BadgeCategory;
  xp_reward: number;
  requirements: string; // JSON
  is_active: boolean;
  created_at: Timestamp;
}

export interface EarnedBadge {
  learner: AgentPubKey;
  badge_id: string;
  earned_at: Timestamp;
  source_hash?: ActionHash;
  metadata?: string;
}

export interface Leaderboard {
  leaderboard_id: string;
  name: string;
  leaderboard_type: LeaderboardType;
  period: LeaderboardPeriod;
  scope: LeaderboardScope;
  entries: LeaderboardEntry[];
  updated_at: Timestamp;
}

export type LeaderboardType = 'TotalXp' | 'WeeklyXp' | 'MonthlyXp' | 'Streak' | 'BadgeCount' | 'Mastery';
export type LeaderboardPeriod = 'Daily' | 'Weekly' | 'Monthly' | 'AllTime';
export type LeaderboardScope = 'Global' | 'Course' | 'Pod' | 'Friends';

export interface LeaderboardEntry {
  rank: number;
  learner: AgentPubKey;
  score: bigint;
  display_name: string;
  change: number; // Rank change from previous period
}

export interface GamificationZomeFunctions {
  get_my_xp: () => Promise<LearnerXp | null>;
  award_xp: (input: { amount: number; activity_type: XpActivityType; source_hash: ActionHash; description: string }) => Promise<ActionHash>;
  get_my_streak: () => Promise<LearnerStreak | null>;
  record_daily_activity: () => Promise<LearnerStreak>;
  use_streak_freeze: () => Promise<LearnerStreak>;
  get_badge_definitions: () => Promise<BadgeDefinition[]>;
  get_my_badges: () => Promise<EarnedBadge[]>;
  award_badge: (input: { badge_id: string; source_hash?: ActionHash }) => Promise<ActionHash>;
  get_leaderboard: (input: { leaderboard_type: LeaderboardType; period: LeaderboardPeriod; scope: LeaderboardScope; limit: number }) => Promise<Leaderboard>;
}

// =============================================================================
// Adaptive Learning Zome Types
// =============================================================================

export type LearningStyle = 'Visual' | 'Auditory' | 'ReadingWriting' | 'Kinesthetic' | 'Multimodal';

export type MasteryLevel = 'Novice' | 'Beginner' | 'Competent' | 'Proficient' | 'Expert' | 'Master';

export type ContentType = 'Video' | 'Text' | 'Interactive' | 'Audio' | 'Quiz' | 'Project' | 'Discussion';

export type RecommendationType = 'NewContent' | 'Review' | 'Practice' | 'Challenge' | 'Remedial' | 'Enrichment';

export type RecommendationReason = 'InZpd' | 'DueForReview' | 'StrengthensWeakness' | 'BuildsOnStrength' | 'LearnerRequested' | 'PrerequisiteMet' | 'TrendingPopular';

export type GoalType = 'SkillMastery' | 'CourseCompletion' | 'DailyPractice' | 'WeeklyHours' | 'StreakDays' | 'BadgeCollection' | 'XpTarget';

export type GoalPriority = 'Low' | 'Medium' | 'High' | 'Critical';

export interface LearnerProfile {
  learner: AgentPubKey;
  primary_style: LearningStyle;
  visual_score_permille: number;
  auditory_score_permille: number;
  reading_score_permille: number;
  kinesthetic_score_permille: number;
  preferred_content_types: ContentType[];
  session_duration_preference_minutes: number;
  daily_goal_minutes: number;
  difficulty_preference_permille: number;
  assessment_count: number;
  last_assessment_at?: Timestamp;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export interface SkillMastery {
  learner: AgentPubKey;
  skill_hash: ActionHash;
  skill_name: string;
  mastery_permille: number;
  level: MasteryLevel;
  // BKT parameters
  learn_rate_permille: number;
  guess_permille: number;
  slip_permille: number;
  // Statistics
  practice_count: number;
  correct_count: number;
  last_practiced_at: Timestamp;
  // ZPD calculation
  zpd_lower_permille: number;
  zpd_upper_permille: number;
  created_at: Timestamp;
  updated_at: Timestamp;
}

export interface Recommendation {
  learner: AgentPubKey;
  recommendation_type: RecommendationType;
  content_hash: ActionHash;
  content_type: ContentType;
  skill_hashes: ActionHash[];
  reason: RecommendationReason;
  confidence_permille: number;
  priority: number;
  estimated_duration_minutes: number;
  zpd_fit_permille: number;
  style_match_permille: number;
  created_at: Timestamp;
  expires_at: Timestamp;
  acted_on: boolean;
  acted_at?: Timestamp;
}

export interface LearningGoal {
  learner: AgentPubKey;
  goal_id: string;
  title: string;
  description: string;
  goal_type: GoalType;
  target_value: number;
  current_value: number;
  priority: GoalPriority;
  deadline?: Timestamp;
  milestones: string[]; // JSON
  is_active: boolean;
  is_completed: boolean;
  completed_at?: Timestamp;
  created_at: Timestamp;
}

export interface SessionAnalytics {
  learner: AgentPubKey;
  session_id: string;
  started_at: Timestamp;
  ended_at?: Timestamp;
  duration_seconds: number;
  activities_completed: number;
  xp_earned: number;
  mastery_changes: Array<{ skill_hash: string; old_permille: number; new_permille: number }>;
  content_types_used: ContentType[];
  avg_difficulty_permille: number;
  engagement_score_permille: number;
  flow_score_permille: number;
  interruptions: number;
}

// Smart Recommendation Types
export interface ContentCandidate {
  content_hash: ActionHash;
  content_type: ContentType;
  skill_hash: ActionHash;
  difficulty_permille: number;
  estimated_minutes: number;
  topic_hash?: ActionHash;
}

export interface SmartScoreComponents {
  flow_score: number;        // ZPD alignment (25% weight)
  style_score: number;       // VARK match (20% weight)
  timing_score: number;      // Circadian bonus (15% weight)
  goal_score: number;        // Goal alignment (25% weight)
  interleaving_score: number; // Topic switching (15% weight)
}

export interface SmartRecommendation {
  content_hash: ActionHash;
  content_type: ContentType;
  skill_hash: ActionHash;
  difficulty_permille: number;
  smart_score_permille: number;
  reasons: RecommendationReason[];
  components: SmartScoreComponents;
}

export interface SmartRecsInput {
  content_pool: ContentCandidate[];
  limit: number;
  hour?: number;           // Current hour (0-23)
  day_of_week?: number;    // Day (0=Sun, 6=Sat)
  recent_topics?: ActionHash[];
  sessions_today?: number;
}

// Flow State Types
export type FlowState =
  | 'Boredom'     // Skill >> Challenge
  | 'Relaxation'  // Skill > Challenge
  | 'Flow'        // Balanced
  | 'Arousal'     // Challenge > Skill
  | 'Anxiety'     // Challenge >> Skill
  | 'Overwhelm'   // Challenge >>> Skill
  | 'Warming'     // Not enough data
  | 'Fatigue';    // Long session, declining performance

export type FlowAdjustment =
  | { IncreaseDifficulty: number }
  | { DecreaseDifficulty: number }
  | { TakeBreak: number }
  | 'SwitchTopic'
  | 'ShortenSession'
  | 'ReduceChallengeScope'
  | 'IntroduceNewMaterial'
  | 'ReviewBasics';

export type FlowTrend = 'Improving' | 'Stable' | 'Declining' | 'Unknown';

export interface FlowMetrics {
  skill_level_permille: number;
  challenge_level_permille: number;
  balance_permille: number;  // 500 = optimal
  engagement_permille: number;
  fatigue_indicator_permille: number;
}

export interface FlowStateAnalysis {
  state: FlowState;
  confidence_permille: number;
  metrics: FlowMetrics;
  adjustments: FlowAdjustment[];
  trend: FlowTrend;
}

export interface AnalyzeFlowInput {
  skill_mastery_permille: number;
  content_difficulty_permille: number;
  session_duration_minutes: number;
  recent_accuracy_permille: number;
  recent_response_times_ms: number[];
}

export interface OptimalLearningWindow {
  recommended_duration_minutes: number;
  optimal_difficulty_range: [number, number];  // [min, max] permille
  best_hours: number[];
  suggested_content_types: ContentType[];
  estimated_energy_permille: number;
  confidence_permille: number;
}

// Caching & Batch Query Types
export interface LearnerStats {
  total_skills: number;
  mastered_count: number;      // >= 800 permille
  in_progress_count: number;   // 200-799 permille
  novice_count: number;        // < 200 permille
  avg_mastery_permille: number;
  total_attempts: number;
  overall_accuracy_permille: number;
  dominant_style: LearningStyle;
}

export interface LearnerContext {
  profile?: LearnerProfile;
  masteries: SkillMastery[];
  goals: LearningGoal[];
  due_for_review: SkillMastery[];
  strengths: SkillMastery[];    // Top 5 by mastery
  weaknesses: SkillMastery[];   // Bottom 5 by mastery
  stats: LearnerStats;
}

export type MasterySortField = 'Mastery' | 'LastAttempt' | 'Attempts' | 'Name';

export interface PaginatedMasteriesInput {
  offset: number;
  limit: number;
  sort_by: MasterySortField;
  ascending: boolean;
  filter_level?: MasteryLevel;
}

export interface PaginatedMasteriesResult {
  masteries: SkillMastery[];
  total_count: number;
  has_more: boolean;
}

// Performance Metrics Types
export interface AlgorithmMetrics {
  bkt_accuracy_permille: number;
  bkt_predictions: number;
  flow_prediction_accuracy_permille: number;
  recommendation_ctr_permille: number;
  avg_followed_smart_score: number;
  avg_skipped_smart_score: number;
}

export interface QueryMetrics {
  avg_context_response_us: number;
  avg_recommendation_response_us: number;
  total_calls: number;
  mastery_cache_hit_permille: number;
  cross_zome_calls: number;
  peak_concurrent_ops: number;
}

export interface EffectivenessMetrics {
  avg_mastery_gain_per_session: number;
  avg_time_to_proficiency_minutes: number;
  retention_rate_permille: number;
  goal_completion_rate_permille: number;
  avg_session_flow_score: number;
  streak_maintenance_rate_permille: number;
}

export interface PerformanceMetrics {
  collected_at: Timestamp;
  period_hours: number;
  algorithm_metrics: AlgorithmMetrics;
  query_metrics: QueryMetrics;
  effectiveness_metrics: EffectivenessMetrics;
}

export interface CollectMetricsInput {
  period_hours?: number;
}

export interface BenchmarkResult {
  operation: string;
  samples: number;
  min_us: number;
  max_us: number;
  avg_us: number;
  p50_us: number;
  p95_us: number;
  p99_us: number;
}

export interface BenchmarkSummary {
  run_at: Timestamp;
  total_duration_us: number;
  benchmarks: BenchmarkResult[];
  recommendations: string[];
}

// =============================================================================
// Retention Prediction Types (Ebbinghaus Forgetting Curve + FSRS)
// =============================================================================

/** Comprehensive retention prediction based on forgetting curve */
export interface RetentionPrediction {
  /** Current predicted retention probability (0-1000 permille) */
  current_retention_permille: number;
  /** Predicted retention in 1 hour (0-1000) */
  retention_1h_permille: number;
  /** Predicted retention in 1 day (0-1000) */
  retention_1d_permille: number;
  /** Predicted retention in 7 days (0-1000) */
  retention_7d_permille: number;
  /** Predicted retention in 30 days (0-1000) */
  retention_30d_permille: number;
  /** Memory stability in minutes (half-life of retention) */
  stability_minutes: number;
  /** Minutes until retention drops below 80% */
  time_to_80_percent_minutes: number;
  /** Minutes until retention drops below 50% */
  time_to_50_percent_minutes: number;
  /** Difficulty factor (1000-5000 range) */
  difficulty_factor_permille: number;
  /** Number of successful reviews (strengthens stability) */
  successful_reviews: number;
  /** Retrievability score (ease of recall, 0-1000) */
  retrievability_permille: number;
}

export interface RetentionPredictionInput {
  skill_hash: ActionHash;
}

/** Individual skill retention with associated hash */
export interface SkillRetentionPrediction {
  skill_hash: ActionHash;
  prediction: RetentionPrediction;
  is_at_risk: boolean;
}

export interface BatchRetentionInput {
  skill_hashes: ActionHash[];
}

/** Batch retention prediction result */
export interface BatchRetentionResult {
  predictions: SkillRetentionPrediction[];
  overall_retention_permille: number;
  skills_at_risk: number;
  skills_healthy: number;
}

export interface ReviewScheduleInput {
  skill_hash: ActionHash;
  target_retention_permille: number;
  forecast_days: number;
}

/** Optimal review schedule to maintain target retention */
export interface ReviewScheduleResult {
  skill_hash: ActionHash;
  current_stability_minutes: number;
  schedule_minutes: number[];
  total_reviews: number;
}

export interface AdaptiveZomeFunctions {
  // Profile Management
  get_my_profile: () => Promise<LearnerProfile | null>;
  update_profile: (profile: Partial<LearnerProfile>) => Promise<ActionHash>;
  submit_style_assessment: (scores: { visual: number; auditory: number; reading: number; kinesthetic: number }) => Promise<LearnerProfile>;

  // Skill Mastery
  get_skill_mastery: (skill_hash: ActionHash) => Promise<SkillMastery | null>;
  get_my_skills: () => Promise<SkillMastery[]>;
  update_skill_mastery: (input: { skill_hash: ActionHash; correct: boolean }) => Promise<SkillMastery>;

  // Basic Recommendations
  get_recommendations: (limit: number) => Promise<Recommendation[]>;
  act_on_recommendation: (recommendation_hash: ActionHash) => Promise<Recommendation>;

  // Smart Recommendations (v2) - Learning science-backed
  generate_smart_recommendations_v2: (input: SmartRecsInput) => Promise<SmartRecommendation[]>;

  // Goals
  get_my_goals: () => Promise<LearningGoal[]>;
  create_goal: (goal: Omit<LearningGoal, 'learner' | 'current_value' | 'is_completed' | 'completed_at' | 'created_at'>) => Promise<ActionHash>;
  update_goal_progress: (goal_id: string, progress: number) => Promise<LearningGoal>;

  // Session Management
  start_session: () => Promise<ActionHash>;
  end_session: (session_hash: ActionHash) => Promise<SessionAnalytics>;
  get_session_analytics: (session_hash: ActionHash) => Promise<SessionAnalytics | null>;

  // Flow State Analysis (Csikszentmihalyi)
  analyze_flow_state: (input: AnalyzeFlowInput) => Promise<FlowStateAnalysis>;
  get_optimal_learning_window: () => Promise<OptimalLearningWindow>;

  // Learner Context Caching
  get_learner_context: () => Promise<LearnerContext>;
  get_masteries_paginated: (input: PaginatedMasteriesInput) => Promise<PaginatedMasteriesResult>;

  // Performance Metrics & Benchmarking
  collect_performance_metrics: (input?: CollectMetricsInput) => Promise<PerformanceMetrics>;
  run_benchmarks: () => Promise<BenchmarkSummary>;

  // Retention Prediction (Ebbinghaus Forgetting Curve + FSRS)
  predict_skill_retention: (input: RetentionPredictionInput) => Promise<RetentionPrediction>;
  predict_retention_batch: (input: BatchRetentionInput) => Promise<BatchRetentionResult>;
  get_optimal_review_schedule: (input: ReviewScheduleInput) => Promise<ReviewScheduleResult>;

  // Metacognition & Confidence Calibration
  calculate_calibration: (judgments: ConfidenceJudgment[]) => Promise<CalibrationMetrics>;
  record_confidence_judgment: (input: RecordJudgmentInput) => Promise<ActionHash>;
  get_calibration_history: () => Promise<CalibrationMetrics[]>;

  // Enhanced Flow State Assessment
  assess_flow_state: (input: FlowAssessmentInput) => Promise<FlowAssessment>;

  // Personalized Learning Path
  generate_learning_path: (input: GeneratePathInput) => Promise<PersonalizedLearningPath>;

  // Peer Learning & Study Groups
  find_peer_matches: (input: PeerMatchInput) => Promise<PeerMatch[]>;
  get_study_groups: (skill_hash?: ActionHash) => Promise<StudyGroup[]>;

  // ======= NEW: 7 Research-Backed Learning Science Features =======

  // Feature 1: Bloom's Taxonomy Cognitive Levels
  get_skill_mastery_by_level: (skill_hash: ActionHash) => Promise<SkillMasteryByLevel>;
  update_cognitive_level: (input: { skill_hash: ActionHash; level: CognitiveLevel; mastery: number }) => Promise<SkillMasteryByLevel>;

  // Feature 2: Transfer of Learning Assessment
  assess_transfer: (input: { skill_hash: ActionHash; target_context: string; transfer_type: TransferType }) => Promise<TransferAssessment>;
  get_transfer_map: (skill_hash: ActionHash) => Promise<TransferMap>;

  // Feature 3: Elaborative Interrogation Prompts
  generate_elaboration_prompt: (input: { skill_hash: ActionHash; cognitive_level: CognitiveLevel }) => Promise<ElaborationPrompt>;
  submit_elaboration: (input: { skill_hash: ActionHash; prompt_type: ElaborationPromptType; response: string }) => Promise<ElaborationResult>;

  // Feature 4: Worked Examples Dynamic Fading
  get_worked_example_recommendation: (input: { skill_hash: ActionHash; mastery_permille: number }) => Promise<WorkedExampleRecommendation>;
  calculate_worked_example_ratio: (mastery_permille: number) => Promise<number>;

  // Feature 5: Expertise Reversal Detection
  detect_expertise_reversal: (input: { mastery_permille: number; content_complexity: number }) => Promise<ExpertiseReversalResult>;
  get_expertise_recommendation: (input: { mastery_permille: number; content_complexity: number }) => Promise<ExpertiseRecommendation>;

  // Feature 6: Desirable Difficulties Framework (Bjork)
  calculate_desirable_difficulties: (input: DesirableDifficultyInput) => Promise<DesirableDifficultyProfile>;

  // Feature 7: Dual Coding Content Pairing (Mayer)
  recommend_dual_coding: (input: DualCodingInput) => Promise<DualCodingRecommendation>;
  calculate_dual_coding_load: (modalities: ContentModality[]) => Promise<number>;
}

// =============================================================================
// Metacognition & Confidence Calibration Types
// =============================================================================

/** A confidence judgment for metacognitive calibration */
export interface ConfidenceJudgment {
  /** Skill or question this judgment is about */
  skill_hash: ActionHash;
  /** Predicted confidence before answering (0-1000) */
  predicted_confidence: number;
  /** Actual performance after answering (0-1000) */
  actual_performance: number;
  /** When this judgment was made */
  judged_at: Timestamp;
}

/** Input for recording a new confidence judgment */
export interface RecordJudgmentInput {
  skill_hash: ActionHash;
  predicted_confidence: number;
  actual_performance: number;
}

/** Calibration trend direction */
export type CalibrationTrend = 'Improving' | 'Stable' | 'Declining';

/** Calibration metrics for metacognitive assessment */
export interface CalibrationMetrics {
  /** Overall calibration score (0-1000, 1000 = perfect) */
  calibration_score: number;
  /** Overconfidence bias (positive = overconfident) */
  overconfidence_bias: number;
  /** Mean absolute error between predictions and outcomes */
  mean_absolute_error: number;
  /** Number of judgments used */
  judgment_count: number;
  /** Brier score for probabilistic accuracy */
  brier_score: number;
  /** Trend over recent judgments */
  calibration_trend: CalibrationTrend;
}

// =============================================================================
// Enhanced Flow State Types
// =============================================================================

/** Flow-based recommendations for adaptive learning */
export type FlowRecommendation =
  | 'IncreaseDifficulty'
  | 'MaintainCurrent'
  | 'DecreaseDifficulty'
  | 'TakeBreak'
  | 'SwitchTopic'
  | 'ReviewBasics';

/** Enhanced flow state assessment result */
export interface FlowAssessment {
  /** Current flow state */
  state: FlowState;
  /** Challenge-skill balance (-1000 to +1000, 0 = perfect) */
  balance: number;
  /** Recommended difficulty adjustment (-500 to +500) */
  difficulty_adjustment: number;
  /** Time in current state (minutes) */
  time_in_state: number;
  /** Engagement score (0-1000) */
  engagement_score: number;
  /** Recommendation for next activity */
  recommendation: FlowRecommendation;
}

/** Input for flow state assessment */
export interface FlowAssessmentInput {
  /** Current skill mastery (0-1000) */
  skill_level: number;
  /** Current challenge difficulty (0-1000) */
  challenge_level: number;
  /** Recent accuracy (0-1000) */
  recent_accuracy: number;
  /** Response time trend (1000 = normal, >1000 = slower) */
  response_time_factor: number;
  /** Minutes in current session */
  session_minutes: number;
  /** Errors in last 5 minutes */
  recent_errors: number;
}

// =============================================================================
// Personalized Learning Path Types
// =============================================================================

/** Reasons for including a step in the learning path */
export type PathStepReason =
  | 'UnlocksOthers'
  | 'DueForReview'
  | 'NearMastery'
  | 'Foundation'
  | 'GoalRelated'
  | 'TimeOptimal'
  | 'FillsGap';

/** A step in a personalized learning path */
export interface LearningPathStep {
  /** Skill to learn */
  skill_hash: ActionHash;
  /** Recommended order (1 = first) */
  order: number;
  /** Estimated time to master (minutes) */
  estimated_minutes: number;
  /** Why this skill is recommended now */
  reason: PathStepReason;
  /** Readiness score (0-1000) */
  readiness: number;
  /** Priority score (0-1000) */
  priority: number;
}

/** A personalized learning path */
export interface PersonalizedLearningPath {
  /** Ordered steps in the path */
  steps: LearningPathStep[];
  /** Total estimated time (minutes) */
  total_estimated_minutes: number;
  /** Skills that will be unlocked */
  skills_to_unlock: number;
  /** Path confidence (0-1000) */
  confidence: number;
  /** When this path was generated */
  generated_at: Timestamp;
}

/** Input for generating a learning path */
export interface GeneratePathInput {
  /** Skills available to learn */
  available_skills: ActionHash[];
  /** Current mastery levels (skill -> permille) */
  current_masteries: Map<string, number>;
  /** Optional goal skill to optimize for */
  goal_skill?: ActionHash;
  /** Maximum time available (minutes) */
  available_minutes: number;
  /** Current hour (0-23) for time-of-day optimization */
  current_hour: number;
}

// =============================================================================
// Peer Learning & Study Group Types
// =============================================================================

/** Reasons for matching with a peer */
export type PeerMatchReason =
  | 'SimilarLevel'
  | 'PeerCanTutor'
  | 'YouCanTutor'
  | 'ComplementarySkills'
  | 'SharedGoals';

/** A potential peer learning match */
export interface PeerMatch {
  /** Peer's agent public key */
  peer_agent: AgentPubKey;
  /** Compatibility score (0-1000) */
  compatibility_score: number;
  /** Reason for the match */
  match_reason: PeerMatchReason;
  /** Skills you both have */
  shared_skills: ActionHash[];
  /** Skills where you complement each other */
  complementary_skills: ActionHash[];
}

/** Input for finding peer matches */
export interface PeerMatchInput {
  /** Skills you want to learn/teach */
  focus_skills: ActionHash[];
  /** Your current mastery levels */
  my_masteries: Map<string, number>;
  /** Prefer peers who can tutor you */
  prefer_tutors: boolean;
  /** Maximum number of matches to return */
  limit: number;
}

/** A study group for collaborative learning */
export interface StudyGroup {
  /** Group identifier */
  group_id: string;
  /** Display name */
  name: string;
  /** Skills this group focuses on */
  focus_skills: ActionHash[];
  /** Number of members */
  member_count: number;
  /** Average mastery of members (0-1000) */
  avg_mastery: number;
  /** Activity level (0-1000) */
  activity_level: number;
}

// =============================================================================
// Integration Zome Types
// =============================================================================

export type LearningEventType =
  | 'SrsReview'
  | 'SrsGraduated'
  | 'LessonComplete'
  | 'QuizPassed'
  | 'QuizFailed'
  | 'ProjectSubmit'
  | 'PeerHelp'
  | 'ContentCreated'
  | 'BadgeEarned'
  | 'SkillMastered'
  | 'GoalAchieved'
  | 'StreakMilestone'
  | 'ChallengeComplete'
  | 'DailyLogin';

export type SourceZome = 'Srs' | 'Gamification' | 'Adaptive' | 'Learning' | 'Credential' | 'Dao' | 'Pods' | 'Knowledge';

export interface LearningEvent {
  learner: AgentPubKey;
  event_type: LearningEventType;
  source_hash: ActionHash;
  source_zome: SourceZome;
  xp_gained: number;
  mastery_change: number; // -1000 to +1000
  skills_affected: ActionHash[];
  quality_permille: number;
  duration_seconds: number;
  streak_day?: number;
  occurred_at: Timestamp;
}

export interface LearnerProgressAggregate {
  learner: AgentPubKey;
  // XP & Level
  total_xp: bigint;
  level: number;
  xp_to_next_level: number;
  // Streaks
  current_streak_days: number;
  longest_streak_days: number;
  streak_bonus_permille: number;
  // SRS Stats
  srs_cards_total: number;
  srs_cards_mature: number;
  srs_cards_learning: number;
  srs_reviews_today: number;
  srs_accuracy_permille: number;
  // Mastery
  skills_tracked: number;
  skills_mastered: number;
  avg_mastery_permille: number;
  skills_due_review: number;
  // Learning Style
  primary_style: LearningStyle;
  style_confidence_permille: number;
  // Goals
  active_goals: number;
  completed_goals: number;
  goal_progress_permille: number;
  // Engagement
  total_sessions: number;
  total_learning_minutes: number;
  avg_session_minutes: number;
  days_active: number;
  // Achievements
  badges_earned: number;
  rare_badges: number;
  leaderboard_rank?: number;
  // Metadata
  confidence_permille: number;
  last_activity_at: Timestamp;
  aggregated_at: Timestamp;
}

export type TriggerMetric =
  | 'TotalXp'
  | 'Level'
  | 'DailyXp'
  | 'CurrentStreak'
  | 'LongestStreak'
  | 'SrsCardsReviewed'
  | 'SrsCardsMatured'
  | 'SrsAccuracy'
  | 'SrsStreakPerfect'
  | 'SkillsMastered'
  | 'AvgMastery'
  | 'MasteryGained'
  | 'GoalsCompleted'
  | 'GoalProgress'
  | 'TotalSessions'
  | 'TotalMinutes'
  | 'DaysActive'
  | 'PeersHelped'
  | 'ContentCreated'
  | 'LikesReceived'
  | 'BadgesEarned'
  | 'LeaderboardRank';

export type ComparisonOp = 'GreaterThan' | 'GreaterOrEqual' | 'LessThan' | 'LessOrEqual' | 'Equal' | 'NotEqual';

export interface TriggerCondition {
  metric: TriggerMetric;
  operator: ComparisonOp;
  target_value: number;
  timeframe_seconds?: number;
}

export interface TriggerReward {
  xp_amount: number;
  badge_id?: string;
  custom_data?: string;
}

export interface AchievementTrigger {
  trigger_id: string;
  name: string;
  description: string;
  conditions: TriggerCondition[];
  reward: TriggerReward;
  repeatable: boolean;
  cooldown_seconds?: number;
  is_active: boolean;
  created_at: Timestamp;
}

export type SessionState = 'Planning' | 'Active' | 'Paused' | 'Completed' | 'Abandoned';

export type PlannedActivityType = 'SrsReview' | 'NewLesson' | 'SkillPractice' | 'QuizAttempt' | 'ProjectWork' | 'Review' | 'Challenge' | 'Break';

export interface PlannedActivity {
  activity_type: PlannedActivityType;
  target_hash: string;
  estimated_minutes: number;
  priority: number; // 1-10 scale (u8 in Rust)
  reason: string;
}

export interface CompletedActivity {
  activity_type: PlannedActivityType;
  target_hash: string;
  duration_seconds: number;
  success: boolean;
  quality_permille: number;
  xp_earned: number;
  completed_at: Timestamp;
}

export interface MasteryChange {
  skill_hash: string;
  old_permille: number;
  new_permille: number;
}

export interface OrchestratedSession {
  learner: AgentPubKey;
  session_id: string;
  planned_activities: PlannedActivity[];
  completed_activities: CompletedActivity[];
  state: SessionState;
  target_minutes: number;
  actual_seconds: number;
  xp_earned: number;
  mastery_changes: MasteryChange[];
  flow_score_permille: number;
  breaks_taken: number;
  started_at: Timestamp;
  ended_at?: Timestamp;
}

// =============================================================================
// Smart Session Planning Types
// =============================================================================

/** Type of session to plan */
export type SessionType =
  | 'Balanced'      // Mix of review, learning, practice
  | 'ReviewFocus'   // Prioritize retention (SRS/review)
  | 'NewLearning'   // Prioritize new content
  | 'Practice'      // Prioritize skill practice
  | 'DeepWork';     // Long focus sessions with fewer breaks

/** Input for smart session planning */
export interface SmartSessionPlanInput {
  /** Target session duration in minutes */
  target_minutes: number;
  /** Focus area (optional) - prioritize specific skill */
  focus_skill?: ActionHash;
  /** Session type preference */
  session_type: SessionType;
  /** Include breaks */
  include_breaks: boolean;
}

/** Type of break */
export type BreakType = 'Micro' | 'Short' | 'Pomodoro' | 'Long';

/** Planned break */
export interface PlannedBreak {
  /** When to take break (minutes into session) */
  after_minutes: number;
  /** Duration in minutes */
  duration_minutes: number;
  /** Break type */
  break_type: BreakType;
  /** Suggestion for break activity */
  suggestion: string;
}

/** A smartly planned activity with reasoning */
export interface SmartPlannedActivity {
  /** Activity type */
  activity_type: PlannedActivityType;
  /** Target skill or content hash */
  target_hash: ActionHash;
  /** Estimated duration in minutes */
  estimated_minutes: number;
  /** Priority (1-10, higher = more important) */
  priority: number;
  /** Why this activity was chosen */
  reasoning: string[];
  /** Expected difficulty match (0-1000) */
  difficulty_match_permille: number;
  /** Estimated XP gain */
  estimated_xp: number;
  /** Order in session */
  sequence: number;
}

/** Smart session plan with detailed activities */
export interface SmartSessionPlan {
  /** Unique plan ID */
  plan_id: string;
  /** Total planned minutes */
  total_minutes: number;
  /** Planned activities in order */
  activities: SmartPlannedActivity[];
  /** Estimated XP gain */
  estimated_xp: number;
  /** Priority skills addressed */
  priority_skills: ActionHash[];
  /** Skills at retention risk */
  at_risk_skills: number;
  /** Recommended breaks */
  breaks: PlannedBreak[];
  /** Flow state optimization notes */
  flow_notes: string;
  /** Confidence in plan quality (0-1000) */
  confidence_permille: number;
  /** Generated timestamp */
  generated_at: Timestamp;
}

export interface DailyLearningReport {
  learner: AgentPubKey;
  date: number; // YYYYMMDD format
  sessions_count: number;
  total_minutes: number;
  events_count: number;
  srs_reviews: number;
  srs_new_cards: number;
  srs_accuracy_permille: number;
  xp_earned: number;
  level_ups: number;
  skills_practiced: number;
  skills_improved: number;
  mastery_gained: number;
  streak_continued: boolean;
  streak_day: number;
  goal_progress: number;
  goals_completed: number;
  badges_earned: string[];
  generated_at: Timestamp;
}

// =============================================================================
// Learning Analytics Types
// =============================================================================

export type AnalyticsPeriod = 'Today' | 'Week' | 'Month' | 'Quarter' | 'Year' | 'AllTime';

export type TrendDirection = 'Up' | 'Down' | 'Stable';

/** Metric with trend analysis showing current vs previous value */
export interface MetricWithTrend {
  /** Current value */
  current_value: number;
  /** Previous period value */
  previous_value: number;
  /** Percentage change from previous */
  change_percent: number;
  /** Trend direction */
  trend: TrendDirection;
}

/** Skill summary for analytics */
export interface SkillSummary {
  /** Skill hash */
  skill_hash: ActionHash;
  /** Skill name */
  skill_name: string;
  /** Mastery level 0-1000 */
  mastery_permille: number;
  /** Retention prediction 0-1000 */
  retention_permille: number;
  /** XP earned on this skill */
  xp_earned: number;
  /** Time spent on skill */
  minutes_practiced: number;
}

/** Comprehensive learning analytics summary */
export interface LearningAnalyticsSummary {
  /** Time period for analytics */
  period: AnalyticsPeriod;
  /** Period start timestamp */
  period_start: Timestamp;
  /** Period end timestamp */
  period_end: Timestamp;

  // Session metrics
  /** Total learning sessions with trend */
  total_sessions: MetricWithTrend;
  /** Total learning minutes with trend */
  total_minutes: MetricWithTrend;
  /** Average session length in minutes */
  avg_session_minutes: number;
  /** Days active with trend */
  active_days: MetricWithTrend;

  // XP & Skills
  /** XP earned with trend */
  xp_earned: MetricWithTrend;
  /** Skills practiced with trend */
  skills_practiced: MetricWithTrend;
  /** Skills mastered with trend */
  skills_mastered: MetricWithTrend;
  /** Average mastery 0-1000 with trend */
  avg_mastery_permille: MetricWithTrend;

  // SRS metrics
  /** SRS reviews with trend */
  srs_reviews: MetricWithTrend;
  /** SRS accuracy 0-1000 with trend */
  srs_accuracy_permille: MetricWithTrend;
  /** Retention rate 0-1000 */
  retention_rate_permille: number;
  /** Cards overdue for review */
  overdue_cards: number;

  // Progress
  /** Current level */
  level: number;
  /** Progress to next level 0-1000 */
  level_progress_permille: number;
  /** Current streak with trend */
  current_streak: MetricWithTrend;
  /** Badges earned with trend */
  badges_earned: MetricWithTrend;
  /** Goals completed with trend */
  goals_completed: MetricWithTrend;

  // Skill breakdowns
  /** Top performing skills */
  top_skills: SkillSummary[];
  /** Skills needing attention */
  weakest_skills: SkillSummary[];

  // Daily breakdown
  /** Daily XP for charting */
  daily_xp: number[];
  /** Daily minutes for charting */
  daily_minutes: number[];

  /** When this report was generated */
  generated_at: Timestamp;
}

/** Weekly trend data point */
export interface WeeklyTrendPoint {
  /** Week number */
  week: number;
  /** Value for this week */
  value: number;
}

/** Learning trends over time */
export interface LearningTrends {
  /** Weekly XP totals */
  weekly_xp: WeeklyTrendPoint[];
  /** Weekly active minutes */
  weekly_minutes: WeeklyTrendPoint[];
  /** Mastery progression (timestamp, avg_mastery) */
  mastery_trend: Array<[Timestamp, number]>;
  /** Streak history (timestamp, streak_days) */
  streak_history: Array<[Timestamp, number]>;
}

/** Input for analytics query */
export interface AnalyticsInput {
  /** Time period to analyze */
  period: AnalyticsPeriod;
  /** Optional specific learner (defaults to self) */
  learner?: AgentPubKey;
}

export interface IntegrationZomeFunctions {
  // Event Management
  record_event: (event: Omit<LearningEvent, 'occurred_at'>) => Promise<ActionHash>;
  get_recent_events: (limit: number) => Promise<LearningEvent[]>;

  // Progress Aggregation
  get_my_progress: () => Promise<LearnerProgressAggregate | null>;
  refresh_progress: () => Promise<LearnerProgressAggregate>;

  // Achievement Triggers
  get_achievement_triggers: () => Promise<AchievementTrigger[]>;
  create_trigger: (trigger: Omit<AchievementTrigger, 'created_at'>) => Promise<ActionHash>;

  // Session Orchestration
  start_orchestrated_session: (input: { target_minutes: number }) => Promise<ActionHash>;
  get_session: (session_hash: ActionHash) => Promise<OrchestratedSession | null>;
  complete_activity: (input: { session_hash: ActionHash; activity: CompletedActivity }) => Promise<OrchestratedSession>;
  end_orchestrated_session: (session_hash: ActionHash) => Promise<OrchestratedSession>;

  // Smart Session Planning
  plan_smart_session: (input: SmartSessionPlanInput) => Promise<SmartSessionPlan>;
  start_session_from_plan: (plan: SmartSessionPlan) => Promise<ActionHash>;

  // Daily Reports
  get_daily_report: (date: number) => Promise<DailyLearningReport | null>;
  generate_daily_report: () => Promise<ActionHash>;

  // Learning Analytics
  get_learning_analytics: (input: AnalyticsInput) => Promise<LearningAnalyticsSummary>;
  get_learning_trends: (weeks: number) => Promise<LearningTrends>;

  // Interleaved Practice
  generate_interleaved_sequence: (input: InterleavedPracticeInput) => Promise<InterleavedPracticeSequence>;

  // Knowledge Graph
  check_skill_readiness: (input: ReadinessCheckInput) => Promise<KnowledgeGraphNode>;

  // Cognitive Load Management
  assess_cognitive_load: (input: CognitiveLoadInput) => Promise<CognitiveState>;

  // Learning Velocity
  calculate_learning_velocity: (input: VelocityInput) => Promise<LearningVelocity>;
}

// =============================================================================
// Interleaved Practice Types (Learning Science)
// =============================================================================

/** Skill categories for interleaving - mixing different types improves retention */
export type SkillCategory = 'Conceptual' | 'Procedural' | 'Factual' | 'Applied' | 'Creative';

/** A skill item for interleaved practice */
export interface InterleavedSkillItem {
  skill_hash: ActionHash;
  category: SkillCategory;
  /** Current mastery 0-1000 */
  mastery_permille: number;
  /** Urgency for review 0-1000 (higher = more urgent) */
  urgency_permille: number;
  /** Difficulty 0-1000 */
  difficulty_permille: number;
  /** Estimated practice time in minutes */
  estimated_minutes: number;
}

/** Input for generating interleaved practice sequence */
export interface InterleavedPracticeInput {
  skills: InterleavedSkillItem[];
  session_minutes: number;
  min_per_category: number;
  max_consecutive: number;
}

/** A practice sequence item with spacing info */
export interface PracticeSequenceItem {
  skill_hash: ActionHash;
  category: SkillCategory;
  position: number;
  /** Spacing from last same-category item */
  spacing_from_last: number;
  /** Is this a "desirable difficulty" item? */
  is_desirable_difficulty: boolean;
}

/** Result of interleaved practice generation */
export interface InterleavedPracticeSequence {
  sequence: PracticeSequenceItem[];
  categories_used: SkillCategory[];
  avg_spacing: number;
  /** Interleaving score 0-1000 (higher = more interleaved) */
  interleaving_score: number;
  total_minutes: number;
  confidence_permille: number;
}

// =============================================================================
// Knowledge Graph Types
// =============================================================================

/** Prerequisite relationship between skills */
export interface SkillPrerequisite {
  skill_hash: ActionHash;
  prerequisite_hash: ActionHash;
  /** Minimum mastery required 0-1000 */
  min_mastery_permille: number;
  is_hard_requirement: boolean;
  /** Weight of this prerequisite 0-1000 */
  weight_permille: number;
}

/** Knowledge graph node representing a skill's position */
export interface KnowledgeGraphNode {
  skill_hash: ActionHash;
  /** Depth in graph (0 = foundational) */
  depth: number;
  prerequisites_count: number;
  dependents_count: number;
  /** Readiness score 0-1000 */
  readiness_permille: number;
  is_unlocked: boolean;
  blocking_skills: ActionHash[];
}

/** Input for checking skill readiness */
export interface ReadinessCheckInput {
  skill_hash: ActionHash;
  prerequisites: SkillPrerequisite[];
  mastery_levels: Array<[ActionHash, number]>;
}

// =============================================================================
// Cognitive Load Management Types
// =============================================================================

/** Cognitive load factors based on Cognitive Load Theory */
export interface CognitiveLoadFactors {
  /** Intrinsic load (inherent difficulty) 0-1000 */
  intrinsic_load: number;
  /** Extraneous load (unnecessary complexity) 0-1000 */
  extraneous_load: number;
  /** Germane load (productive learning effort) 0-1000 */
  germane_load: number;
  /** Time pressure factor 0-1000 */
  time_pressure: number;
  /** Fatigue estimate 0-1000 */
  fatigue_estimate: number;
}

/** Recommendations for cognitive load management */
export type CognitiveRecommendation =
  | 'Continue'
  | 'ReduceDifficulty'
  | 'ShortBreak'
  | 'LongBreak'
  | 'SwitchToEasier'
  | 'EndSessionSoon'
  | 'ImmediateRest';

/** Current cognitive state */
export interface CognitiveState {
  /** Current total load 0-1000 */
  current_load: number;
  /** Personal capacity 0-1000 */
  capacity: number;
  remaining_capacity: number;
  is_overloaded: boolean;
  recommendation: CognitiveRecommendation;
  minutes_until_break: number;
}

/** Input for cognitive load assessment */
export interface CognitiveLoadInput {
  factors: CognitiveLoadFactors;
  session_minutes: number;
  recent_errors: number;
  /** Response time trend (1000 = normal, >1000 = slower) */
  response_time_trend_permille: number;
  /** Personal capacity modifier */
  capacity_modifier_permille: number;
}

// =============================================================================
// Learning Velocity Types
// =============================================================================

/** Learning velocity metrics for pace tracking */
export interface LearningVelocity {
  skills_per_week: number;
  xp_per_hour: number;
  avg_time_to_mastery_mins: number;
  /** Retention after 7 days 0-1000 */
  retention_7d_permille: number;
  /** Velocity trend (positive = improving) */
  velocity_trend_permille: number;
  /** Comparison to cohort (1000 = average) */
  cohort_comparison_permille: number;
  days_to_next_level: number;
}

/** Input for velocity calculation */
export interface VelocityInput {
  skills_mastered_30d: number;
  xp_earned_30d: number;
  learning_minutes_30d: number;
  skills_mastered_prev_30d: number;
  avg_mastery_time_mins: number;
  retention_samples: number[];
  cohort_avg_skills_per_week: number;
  xp_to_next_level: number;
}

// =============================================================================
// FEATURE 1: Bloom's Taxonomy Cognitive Levels
// =============================================================================

/** Bloom's Taxonomy cognitive levels (Remember → Create) */
export type CognitiveLevel =
  | 'Remember'    // Level 1: Recall facts and basic concepts
  | 'Understand'  // Level 2: Explain ideas or concepts
  | 'Apply'       // Level 3: Use information in new situations
  | 'Analyze'     // Level 4: Draw connections among ideas
  | 'Evaluate'    // Level 5: Justify a decision or course of action
  | 'Create';     // Level 6: Produce new or original work

/** Skill mastery broken down by Bloom's cognitive level */
export interface SkillMasteryByLevel {
  skill_hash: ActionHash;
  /** Mastery at Remember level (0-1000) */
  remember_mastery: number;
  /** Mastery at Understand level (0-1000) */
  understand_mastery: number;
  /** Mastery at Apply level (0-1000) */
  apply_mastery: number;
  /** Mastery at Analyze level (0-1000) */
  analyze_mastery: number;
  /** Mastery at Evaluate level (0-1000) */
  evaluate_mastery: number;
  /** Mastery at Create level (0-1000) */
  create_mastery: number;
  /** Current cognitive level achieved */
  current_level: CognitiveLevel;
  /** Composite mastery weighted by level (0-1000) */
  composite_mastery: number;
}

/** Threshold for unlocking next cognitive level (850/1000) */
export const COGNITIVE_UNLOCK_THRESHOLD = 850;

// =============================================================================
// FEATURE 2: Transfer of Learning Assessment
// =============================================================================

/** Types of learning transfer by distance */
export type TransferType =
  | 'NearTransfer'         // Apply to similar context
  | 'IntermediateTransfer' // Apply to related domain
  | 'FarTransfer';         // Apply to completely different domain

/** Transfer assessment for a skill */
export interface TransferAssessment {
  /** Source skill being transferred */
  source_skill: ActionHash;
  /** Target context/skill */
  target_context: string;
  /** Type of transfer attempted */
  transfer_type: TransferType;
  /** Success rate (0-1000) */
  success_permille: number;
  /** When assessed */
  assessed_at: number;
}

/** Transfer map showing how well a skill transfers to different contexts */
export interface TransferMap {
  source_skill: ActionHash;
  /** Original mastery (0-1000) */
  original_mastery: number;
  /** Near transfer success rate (0-1000) */
  near_transfer_success: number;
  /** Intermediate transfer success rate (0-1000) */
  intermediate_transfer_success: number;
  /** Far transfer success rate (0-1000) */
  far_transfer_success: number;
  /** Has deep mastery? (all transfer types >= 700) */
  is_deep_mastery: boolean;
}

// =============================================================================
// FEATURE 3: Elaborative Interrogation Prompts
// =============================================================================

/** Types of elaborative interrogation prompts (increase learning 28-35%) */
export type ElaborationPromptType =
  | 'WhyWorks'             // "Why does this work?"
  | 'HowKnow'              // "How do you know this is true?"
  | 'ExplainSimply'        // "Explain this to a 5-year-old"
  | 'CounterExample'       // "What would disprove this?"
  | 'Connection'           // "How does this connect to X?"
  | 'RealWorldApplication' // "Give a real-world example"
  | 'ProcessSteps'         // "Walk through the steps"
  | 'UnderlyingPrinciple'; // "What's the underlying principle?"

/** An elaboration prompt for self-explanation */
export interface ElaborationPrompt {
  skill_hash: ActionHash;
  prompt_type: ElaborationPromptType;
  prompt_text: string;
  expected_answer_elements: string[];
  difficulty_permille: number;
}

/** Result of an elaboration attempt */
export interface ElaborationResult {
  /** Quality of elaboration (0-1000) */
  quality_score: number;
  /** Should extend review interval? */
  extend_interval: boolean;
  /** Interval multiplier (1000 = 1x, 1500 = 1.5x) */
  interval_multiplier: number;
  /** Should trigger retrieval practice? */
  trigger_retrieval_practice: boolean;
  /** Feedback message */
  feedback: string;
}

// =============================================================================
// FEATURE 4: Worked Examples Dynamic Fading
// =============================================================================

/** Content format for worked examples fading */
export type ContentFormat =
  | 'FullWorkedExample'
  | { PartialWorkedExample: { steps_shown_permille: number } }
  | { GuidedProblem: { difficulty: number } }
  | { IndependentProblem: { difficulty: number } };

/** Recommendation for worked example usage */
export interface WorkedExampleRecommendation {
  /** Recommended format */
  format: ContentFormat;
  /** Worked example ratio (0-1000, e.g., 800 = 80% examples) */
  example_ratio_permille: number;
  /** Steps to show if partial */
  steps_to_show: number;
  /** Total steps in problem */
  total_steps: number;
  /** Reasoning for recommendation */
  reasoning: string;
}

// =============================================================================
// FEATURE 5: Expertise Reversal Detection
// =============================================================================

/** Expertise reversal detection result */
export interface ExpertiseReversalResult {
  /** Current mastery level (0-1000) */
  mastery_permille: number;
  /** Content complexity (0-1000) */
  content_complexity: number;
  /** Is expertise reversal likely? */
  reversal_detected: boolean;
  /** Potential learning decrease percentage */
  potential_decrease_percent: number;
}

/** Recommendation for content based on expertise */
export type ExpertiseRecommendation =
  | 'ContentAppropriate'   // Content matches expertise
  | 'IncreaseChallenge'    // Expert needs harder content
  | 'SwitchToExpertVersion' // Detailed explanations hurting expert
  | 'SimplifyContent'      // Content too complex for novice
  | 'AddPrerequisites';    // Missing foundational knowledge

// =============================================================================
// FEATURE 6: Desirable Difficulties Framework (Bjork)
// =============================================================================

/** Retention goal types (not to be confused with learning objectives) */
export type RetentionGoal =
  | 'ShortTermRetention'   // Quick mastery for exam
  | 'LongTermRetention'    // Long-term skill building (default)
  | 'Transfer'             // Ability to use in new situations
  | 'DeepUnderstanding';   // Deep conceptual understanding

/** Desirable difficulty dimensions based on Bjork's research */
export type DifficultyDimension =
  | 'Spacing'       // Time since last review
  | 'Interleaving'  // Mixing topics within session
  | 'Variability'   // Varying problem contexts
  | 'Transfer'      // Applying in distant contexts
  | 'Generation'    // Generating answers vs. recognition
  | 'Testing';      // Testing rather than restudying

/** Desirable difficulty profile for a learner */
export interface DesirableDifficultyProfile {
  spacing_score: number;
  interleaving_score: number;
  variability_score: number;
  transfer_score: number;
  generation_score: number;
  testing_score: number;
  overall_difficulty_level: number;
  recommended_session: SessionComposition;
}

/** Session composition for desirable difficulties */
export interface SessionComposition {
  /** Percentage of spaced items (0-1000) */
  spaced_items_permille: number;
  /** Percentage of interleaved topics (0-1000) */
  interleaved_permille: number;
  /** Percentage of varied contexts (0-1000) */
  variability_permille: number;
  /** Percentage of transfer practice (0-1000) */
  transfer_permille: number;
  /** Number of topics to mix */
  topics_to_mix: number;
  /** Context variation level (0-1000) */
  context_variation: number;
}

/** Input for calculating desirable difficulties */
export interface DesirableDifficultyInput {
  learner_mastery_avg: number;
  learning_style: LearningStyle;
  retention_goal: RetentionGoal;
  days_since_last_practice: number;
  topics_practiced_last_session: number;
  contexts_used_last_week: number;
}

// =============================================================================
// FEATURE 7: Dual Coding Content Pairing (Mayer)
// =============================================================================

/** Content modality types for dual coding */
export type ContentModality =
  | 'Visual'      // Diagrams, images, videos
  | 'Verbal'      // Text, audio narration
  | 'Kinesthetic' // Interactive, hands-on
  | 'Symbolic'    // Formulas, notation, code
  | 'DualCoded';  // Already combined visual+verbal

/** Dual-coded content entry */
export interface DualCodedContent {
  content_hash: ActionHash;
  primary_modality: ContentModality;
  complementary_modality: ContentModality;
  visual_representation?: string;
  verbal_explanation?: string;
  integration_quality: number;
  cognitive_load_permille: number;
}

/** Recommendation for dual coding */
export interface DualCodingRecommendation {
  primary_modality: ContentModality;
  complementary_modality: ContentModality;
  pairing_rationale: string;
  missing_modality?: ContentModality;
  expected_improvement_permille: number;
}

/** Input for dual coding recommendation */
export interface DualCodingInput {
  learning_style: LearningStyle;
  cognitive_level: CognitiveLevel;
  content_type: string;
  current_modalities: ContentModality[];
}

// =============================================================================
// FEATURE 8: Testing Effect / Retrieval Practice (Roediger & Karpicke)
// 50-70% better retention than restudying
// =============================================================================

/** Types of retrieval practice (ordered by difficulty) */
export type RetrievalType =
  | 'FreeRecall'    // Most difficult - recall without cues
  | 'CuedRecall'    // Some prompts provided
  | 'Recognition'   // Multiple choice
  | 'ShortAnswer'   // Brief written response
  | 'FillInBlank';  // Complete the sentence

/** Schedule for spaced retrieval practice */
export interface RetrievalSchedule {
  skill_hash: ActionHash;
  current_interval_minutes: number;
  successful_retrievals: number;
  failed_retrievals: number;
  optimal_retrieval_type: RetrievalType;
  next_retrieval_due: number;
}

/** Study recommendation based on testing effect research */
export type StudyRecommendation =
  | { type: 'RetrievalPractice'; retrieval_type: RetrievalType }
  | { type: 'Restudy' }
  | { type: 'InterleavedTestStudy'; test_ratio_permille: number };

/** Input for retrieval practice recommendation */
export interface RetrievalPracticeInput {
  skill_hash: ActionHash;
  mastery_permille: number;
  last_retrieval_success: boolean;
  retrievals_today: number;
  time_since_last_retrieval_minutes: number;
}

/** Result of retrieval practice analysis */
export interface RetrievalPracticeResult {
  recommendation: StudyRecommendation;
  schedule: RetrievalSchedule;
  expected_retention_boost_permille: number;
  retrieval_difficulty_progression: RetrievalType[];
}

// =============================================================================
// FEATURE 9: Hypercorrection Effect (Butterfield & Metcalfe)
// 86% vs 64% correction rate for high-confidence errors
// =============================================================================

/** Record of high-confidence errors for hypercorrection */
export interface HighConfidenceError {
  skill_hash: ActionHash;
  confidence_when_wrong: number;
  correct_answer_hash: ActionHash;
  times_corrected: number;
  last_corrected_timestamp: number;
}

/** Feedback intensity levels based on confidence */
export type FeedbackIntensity = 'Light' | 'Standard' | 'Emphasized' | 'Deep';

/** Input for hypercorrection analysis */
export interface HypercorrectionInput {
  skill_hash: ActionHash;
  confidence_permille: number;
  was_correct: boolean;
  prior_high_confidence_errors: number;
}

/** Analysis result for hypercorrection effect */
export interface HypercorrectionAnalysis {
  is_hypercorrection_candidate: boolean;
  surprise_signal_permille: number;
  recommended_feedback: FeedbackIntensity;
  expected_correction_rate_permille: number;
  schedule_immediate_retest: boolean;
}

// =============================================================================
// FEATURE 10: Pre-Testing Effect (Richland, Kornell & Kao)
// 10-25% improvement when testing before learning
// =============================================================================

/** Analysis of whether to use pre-testing */
export interface PreTestAnalysis {
  should_pre_test: boolean;
  expected_boost_permille: number;
  optimal_question_count: number;
  attention_priming_permille: number;
  prior_knowledge_check: boolean;
}

/** Input for pre-test analysis */
export interface PreTestInput {
  skill_hash: ActionHash;
  is_novel_material: boolean;
  learner_curiosity_permille: number;
  time_until_learning_minutes: number;
}

// =============================================================================
// FEATURE 11: Productive Failure Framework (Kapur)
// 20-30% improvement in transfer through structured struggle
// =============================================================================

/** Phases of productive failure */
export type ProductiveFailurePhase =
  | 'Exploration'   // Initial free exploration
  | 'Struggle'      // Attempting without instruction
  | 'Consolidation' // Receive instruction after struggle
  | 'Application';  // Apply learned concepts

/** Metrics for measuring productive struggle */
export interface StruggleMetrics {
  struggle_duration_minutes: number;
  solution_attempts: number;
  approaches_tried: number;
  partial_insights_gained: number;
  frustration_level_permille: number;
}

/** Recommendations during productive failure */
export type ProductiveFailureRecommendation =
  | 'ContinueStruggle'      // Keep exploring
  | 'ProvideHint'           // Give partial scaffold
  | 'BeginConsolidation'    // Move to instruction
  | 'SimplifyProblem'       // Problem too hard
  | 'UseDirectInstruction'; // Skip productive failure

/** Input for productive failure analysis */
export interface ProductiveFailureInput {
  skill_hash: ActionHash;
  struggle_metrics: StruggleMetrics;
  current_phase: ProductiveFailurePhase;
  learner_persistence_permille: number;
}

/** Analysis result for productive failure */
export interface ProductiveFailureAnalysis {
  current_phase: ProductiveFailurePhase;
  recommendation: ProductiveFailureRecommendation;
  optimal_struggle_remaining_minutes: number;
  transfer_benefit_permille: number;
  readiness_for_instruction_permille: number;
}

// =============================================================================
// FEATURE 12: Self-Determination Theory (Deci & Ryan)
// Autonomy, Competence, Relatedness for intrinsic motivation
// =============================================================================

/** The three basic psychological needs */
export interface SDTNeeds {
  autonomy_permille: number;    // Sense of choice and control
  competence_permille: number;  // Feeling effective
  relatedness_permille: number; // Connection with others
}

/** Which need to target */
export type SDTNeedType = 'Autonomy' | 'Competence' | 'Relatedness';

/** Recommendations to enhance SDT needs */
export type SDTRecommendation =
  | 'IncreaseChoices'           // For autonomy
  | 'ReduceControllingLanguage' // For autonomy
  | 'ProvideCompetenceFeedback' // For competence
  | 'AdjustDifficulty'          // For competence
  | 'EncouragePeerInteraction'  // For relatedness
  | 'JoinStudyGroup'            // For relatedness
  | 'AcknowledgeFeelings'       // For autonomy
  | 'ProvideRationale';         // For autonomy

/** Input for SDT assessment */
export interface SDTInput {
  choices_offered_today: number;
  success_rate_recent_permille: number;
  peer_interactions_today: number;
  controlling_language_exposure: number;
}

/** SDT assessment result */
export interface SDTAssessment {
  needs: SDTNeeds;
  lowest_need: SDTNeedType;
  motivation_index_permille: number;
  recommendations: SDTRecommendation[];
  intrinsic_motivation_likely: boolean;
}

// =============================================================================
// FEATURE 13: Growth Mindset Integration (Dweck)
// Fixed vs Growth mindset detection and intervention
// =============================================================================

/** Mindset indicators */
export type MindsetIndicator = 'Fixed' | 'Growth' | 'Mixed';

/** Interventions to promote growth mindset */
export type MindsetIntervention =
  | 'PraiseEffort'        // Focus on process, not ability
  | 'NormalizeStruggle'   // Struggle is part of learning
  | 'TeachPlasticity'     // Brain can grow
  | 'ReframeFailure'      // Failure as feedback
  | 'ShowGrowthExamples'  // Models of growth
  | 'UseYetFraming'       // "Not yet" instead of "can't"
  | 'ChallengeComfortZone'; // Encourage stretching

/** Signals indicating mindset type */
export interface MindsetSignals {
  avoids_challenges_permille: number;
  gives_up_early_permille: number;
  ignores_feedback_permille: number;
  threatened_by_others_success: number;
  embraces_challenges_permille: number;
  persists_through_difficulty_permille: number;
  learns_from_feedback_permille: number;
  inspired_by_others_success: number;
}

/** Input for mindset assessment */
export interface MindsetInput {
  signals: MindsetSignals;
  domain?: string; // Mindset can vary by domain
  recent_failure_response?: string;
}

/** Mindset assessment result */
export interface MindsetAssessment {
  mindset: MindsetIndicator;
  growth_score_permille: number;
  fixed_score_permille: number;
  interventions: MindsetIntervention[];
  domain_specific_patterns: Record<string, MindsetIndicator>;
}

// =============================================================================
// FEATURE 14: Attention/Mind-Wandering Detection
// Response pattern analysis for engagement
// =============================================================================

/** Current attention state */
export type AttentionState =
  | 'Focused'       // Fully engaged
  | 'Drifting'      // Starting to wander
  | 'MindWandering' // Off-task thoughts
  | 'Disengaged'    // Not paying attention
  | 'Guessing'      // Random responses
  | 'Unknown';      // Insufficient data

/** Actions to re-engage attention */
export type ReEngagementAction =
  | 'Continue'              // No action needed
  | 'ThoughtProbe'          // Ask about thoughts
  | 'SwitchContent'         // Change activity
  | 'TakeBreak'             // Rest period
  | 'IncreaseInteractivity'; // More engagement

/** Response time patterns for attention detection */
export interface ResponsePatterns {
  response_times_ms: number[];
  accuracy_sequence: number[];
  variance_permille: number;
  consistency_permille: number;
}

/** Input for attention assessment */
export interface AttentionInput {
  patterns: ResponsePatterns;
  session_duration_minutes: number;
  last_break_minutes_ago: number;
  time_of_day_hour: number;
}

/** Attention assessment result */
export interface AttentionAssessment {
  state: AttentionState;
  confidence_permille: number;
  focus_duration_minutes: number;
  recommended_action: ReEngagementAction;
  predicted_time_until_drift_minutes: number;
}

// =============================================================================
// FEATURE 15: Critical Thinking Framework
// Argument analysis, logical reasoning, and fallacy detection
// =============================================================================

/** Types of claims in an argument */
export type ClaimType =
  | 'Factual'       // Can be verified with evidence
  | 'Evaluative'    // Value judgment
  | 'Policy'        // Proposes action
  | 'Interpretive'  // Explains meaning
  | 'Definitional'  // Defines a concept
  | 'Causal';       // Claims cause-effect relationship

/** Types of evidence supporting claims */
export type EvidenceType =
  | 'Statistical'   // Data and numbers
  | 'Anecdotal'     // Personal stories
  | 'Expert'        // Expert testimony
  | 'Empirical'     // Scientific research
  | 'Logical'       // Reasoning-based
  | 'Analogical'    // Comparison-based
  | 'Historical'    // Historical precedent
  | 'None';         // No evidence provided

/** Strength of evidence (0-1000 permille) */
export type EvidenceStrength = number;

/** Logical fallacies for detection (24 types) */
export type LogicalFallacy =
  // Fallacies of Relevance
  | 'AdHominem'           // Attack the person, not the argument
  | 'AppealToAuthority'   // Inappropriate authority citation
  | 'AppealToEmotion'     // Emotional manipulation
  | 'AppealToTradition'   // "We've always done it this way"
  | 'AppealToNature'      // "Natural = good"
  | 'AppealToPopularity'  // Bandwagon
  | 'RedHerring'          // Irrelevant distraction
  | 'StrawMan'            // Misrepresenting opponent's argument
  // Fallacies of Ambiguity
  | 'Equivocation'        // Shifting word meaning
  | 'Amphiboly'           // Grammatical ambiguity
  // Fallacies of Presumption
  | 'FalseDialemma'       // Only 2 options when more exist
  | 'SlipperySlope'       // Unwarranted chain of consequences
  | 'CircularReasoning'   // Conclusion in premises
  | 'HastyGeneralization' // Too small sample
  | 'FalseCause'          // Post hoc ergo propter hoc
  // Fallacies of Weak Induction
  | 'WeakAnalogy'         // Poor comparison
  | 'AppealToIgnorance'   // No evidence = false
  // Formal Fallacies
  | 'AffirmingConsequent' // If P then Q; Q; therefore P
  | 'DenyingAntecedent'   // If P then Q; not P; therefore not Q
  // Other Common Fallacies
  | 'NoTrueScotsman'      // Arbitrary redefinition
  | 'MovingGoalposts'     // Changing criteria after fact
  | 'TuQuoque'            // "You do it too"
  | 'GeneticFallacy'      // Origin determines value
  | 'Whataboutism';       // Deflection to other issues

/** Argument component for analysis */
export interface ArgumentComponent {
  claim: string;
  claim_type: ClaimType;
  evidence_type: EvidenceType;
  evidence_strength: EvidenceStrength;
  assumptions: string[];
  detected_fallacies: LogicalFallacy[];
}

/** Detected fallacy with confidence */
export interface DetectedFallacy {
  fallacy_type: LogicalFallacy;
  confidence_permille: number;
  explanation: string;
  text_span?: string;
}

/** Full argument analysis result */
export interface ArgumentAnalysis {
  components: ArgumentComponent[];
  overall_strength_permille: number;
  detected_fallacies: DetectedFallacy[];
  missing_elements: string[];
  suggestions: string[];
}

/** Input for argument analysis */
export interface ArgumentInput {
  argument_text: string;
  context?: string;
}

// =============================================================================
// FEATURE 16: Epistemic Vigilance
// Source credibility and cognitive bias detection
// =============================================================================

/** Cognitive biases for detection (15 types) */
export type CognitiveBias =
  | 'ConfirmationBias'       // Seeking confirming info
  | 'AnchoringBias'          // Over-relying on first info
  | 'AvailabilityHeuristic'  // Overweighting recent/vivid
  | 'DunningKruger'          // Overconfidence in low skill
  | 'HindsightBias'          // "Knew it all along"
  | 'SunkCostFallacy'        // Continuing due to investment
  | 'BandwagonEffect'        // Following the crowd
  | 'HaloEffect'             // One trait influences others
  | 'NegativeBias'           // Overweighting negatives
  | 'OptimismBias'           // Overestimating positives
  | 'StatusQuoBias'          // Preferring current state
  | 'Groupthink'             // Conformity over critical thinking
  | 'BlindSpotBias'          // Seeing biases in others, not self
  | 'ProjectionBias'         // Assuming others think like us
  | 'RecencyBias';           // Overweighting recent events

/** Types of information sources */
export type SourceType =
  | 'PeerReviewedJournal'
  | 'ExpertOpinion'
  | 'GovernmentSource'
  | 'NewsMedia'
  | 'SocialMedia'
  | 'PersonalBlog'
  | 'WikiSource'
  | 'AnonymousSource'
  | 'PrimarySource'
  | 'SecondarySource';

/** Source credibility assessment */
export interface SourceCredibility {
  source_type: SourceType;
  credibility_permille: number;
  expertise_relevance_permille: number;
  potential_biases: CognitiveBias[];
  verification_suggestions: string[];
}

/** Detected bias with confidence */
export interface DetectedBias {
  bias_type: CognitiveBias;
  confidence_permille: number;
  evidence: string;
  mitigation_suggestion: string;
}

/** Bias analysis result */
export interface BiasAnalysis {
  detected_biases: DetectedBias[];
  overall_objectivity_permille: number;
  recommendations: string[];
}

/** Input for source evaluation */
export interface SourceEvaluationInput {
  source_url?: string;
  source_type: SourceType;
  author_credentials?: string;
  publication_date?: Timestamp;
  has_citations: boolean;
  peer_reviewed: boolean;
}

/** Input for bias detection */
export interface BiasDetectionInput {
  text: string;
  context?: string;
  author_background?: string;
}

// =============================================================================
// FEATURE 17: Socratic Dialogue
// Probing questions and steel-manning for deeper understanding
// =============================================================================

/** Types of Socratic questions */
export type SocraticQuestionType =
  | 'Clarification'        // "What do you mean by...?"
  | 'ProbeAssumptions'     // "What are you assuming?"
  | 'ProbeEvidence'        // "How do you know this?"
  | 'ProbeViewpoints'      // "What might others think?"
  | 'ProbeImplications'    // "What are the consequences?"
  | 'QuestionTheQuestion'; // "Why is this question important?"

/** A Socratic dialogue entry */
export interface SocraticDialogue {
  question_type: SocraticQuestionType;
  question_text: string;
  depth_level: number;
  probing_target: string;
  expected_thinking: string;
}

/** Steel-manned argument (strongest form of opponent's position) */
export interface SteelMannedArgument {
  original_argument: string;
  steel_manned_version: string;
  strength_improvement_permille: number;
  key_strengths: string[];
}

/** Input for Socratic question generation */
export interface SocraticInput {
  statement: string;
  topic?: string;
  current_depth: number;
  previous_questions?: string[];
}

// =============================================================================
// FEATURE 18: Metacognition Framework
// Thinking about thinking for self-regulated learning
// =============================================================================

/** Metacognitive skills */
export type MetacognitiveSkill =
  | 'Planning'          // Setting goals and strategies
  | 'Monitoring'        // Tracking comprehension
  | 'Evaluating'        // Assessing learning outcomes
  | 'StrategySelection' // Choosing appropriate methods
  | 'SelfExplanation'   // Explaining to oneself
  | 'Debugging';        // Identifying and fixing errors

/** Metacognitive awareness levels */
export type MetacognitiveLevel =
  | 'Tacit'       // Unaware of thinking processes
  | 'Aware'       // Basic awareness of thinking
  | 'Strategic'   // Can choose and apply strategies
  | 'Reflective'; // Deep reflection and self-improvement

/** Learning strategies for metacognition */
export type LearningStrategy =
  | 'Elaboration'
  | 'Organization'
  | 'CriticalThinking'
  | 'Rehearsal'
  | 'Regulation'
  | 'Monitoring';

/** Metacognitive assessment result */
export interface MetacognitiveAssessment {
  current_level: MetacognitiveLevel;
  skill_scores: Record<MetacognitiveSkill, number>;
  prediction_accuracy_permille: number;
  strategy_effectiveness: Record<LearningStrategy, number>;
  recommendations: string[];
}

/** Input for metacognitive assessment */
export interface MetacognitiveInput {
  predicted_performance: number;
  actual_performance: number;
  strategies_used: LearningStrategy[];
  self_reflection_depth: number;
  planning_quality: number;
}

// =============================================================================
// FEATURE 19: Collaborative Knowledge Building
// Perspective-taking and collective intelligence
// =============================================================================

/** Roles in collaborative learning */
export type CollaborationRole =
  | 'Proposer'     // Introduces new ideas
  | 'Questioner'   // Asks clarifying questions
  | 'Challenger'   // Provides counter-arguments
  | 'Synthesizer'  // Combines different viewpoints
  | 'Summarizer'   // Consolidates discussion
  | 'Facilitator'; // Guides the conversation

/** Moves in argumentative discourse */
export type ArgumentationMove =
  | 'Claim'      // Making an assertion
  | 'Support'    // Providing evidence
  | 'Challenge'  // Questioning a claim
  | 'Concede'    // Acknowledging opponent's point
  | 'Qualify'    // Adding conditions
  | 'Synthesize' // Combining viewpoints
  | 'Clarify'    // Explaining further
  | 'Redirect';  // Changing discussion direction

/** Perspective-taking assessment */
export interface PerspectiveAssessment {
  own_perspective_clarity: number;
  other_perspectives_identified: number;
  synthesis_quality_permille: number;
  empathy_score_permille: number;
}

/** Collaborative contribution quality */
export interface CollaborativeContribution {
  role: CollaborationRole;
  moves: ArgumentationMove[];
  builds_on_others_permille: number;
  introduces_novelty_permille: number;
  evidence_quality_permille: number;
}

/** Collaborative assessment result */
export interface CollaborativeAssessment {
  contribution_quality_permille: number;
  perspective_assessment: PerspectiveAssessment;
  role_distribution: Record<CollaborationRole, number>;
  knowledge_building_score: number;
  recommendations: string[];
}

/** Input for collaborative assessment */
export interface CollaborativeInput {
  contributions: CollaborativeContribution[];
  group_size: number;
  discussion_topic: string;
}

// =============================================================================
// FEATURE 20: Creativity & Divergent Thinking
// Torrance dimensions and creative techniques
// =============================================================================

/** Types of creative thinking */
export type CreativeThinkingType =
  | 'Divergent'     // Generating many ideas
  | 'Convergent'    // Finding best solution
  | 'Lateral'       // Indirect approaches
  | 'Analogical'    // Using analogies
  | 'Combinatorial'; // Combining existing ideas

/** Creative thinking techniques */
export type CreativeTechnique =
  | 'Brainstorming'
  | 'SCAMPER'           // Substitute, Combine, Adapt, etc.
  | 'SixThinkingHats'
  | 'MindMapping'
  | 'Analogies'
  | 'ConstraintRemoval'
  | 'RandomStimulus'
  | 'Reversal'
  | 'WhatIf';

/** Torrance creativity dimensions */
export interface TorranceDimensions {
  fluency: number;       // Number of ideas
  flexibility: number;   // Variety of categories
  originality: number;   // Uniqueness of ideas
  elaboration: number;   // Detail and development
}

/** Creativity assessment result */
export interface CreativityAssessment {
  thinking_type: CreativeThinkingType;
  torrance_scores: TorranceDimensions;
  composite_creativity_permille: number;
  technique_effectiveness: Record<CreativeTechnique, number>;
  breakthrough_potential_permille: number;
  recommendations: string[];
}

/** Input for creativity assessment */
export interface CreativityInput {
  ideas: string[];
  categories_used: string[];
  techniques_applied: CreativeTechnique[];
  time_spent_minutes: number;
  constraints?: string[];
}

// =============================================================================
// FEATURE 21: Inquiry-Based Learning
// Question generation and hypothesis formation
// =============================================================================

/** Quality levels of questions */
export type QuestionQuality =
  | 'Factual'       // Simple recall
  | 'Conceptual'    // Understanding relationships
  | 'Procedural'    // How to do something
  | 'Metacognitive' // About learning process
  | 'Generative';   // Creates new knowledge

/** Phases of inquiry-based learning */
export type InquiryPhase =
  | 'Questioning'    // Formulating questions
  | 'Hypothesizing'  // Making predictions
  | 'Investigating'  // Gathering data
  | 'Analyzing'      // Interpreting results
  | 'Concluding'     // Drawing conclusions
  | 'Reflecting';    // Evaluating the process

/** Quality assessment of a hypothesis */
export interface HypothesisQuality {
  testable_permille: number;
  specific_permille: number;
  evidence_based_permille: number;
  falsifiable_permille: number;
}

/** Investigation rigor assessment */
export interface InvestigationRigor {
  controlled_variables_permille: number;
  sample_adequacy_permille: number;
  systematic_approach_permille: number;
  bias_mitigation_permille: number;
}

/** Conclusion validity assessment */
export interface ConclusionValidity {
  matches_evidence_permille: number;
  acknowledges_limitations_permille: number;
  avoids_overgeneralization_permille: number;
  suggests_further_inquiry_permille: number;
}

/** Inquiry assessment result */
export interface InquiryAssessment {
  current_phase: InquiryPhase;
  question_quality: QuestionQuality;
  hypothesis_quality?: HypothesisQuality;
  investigation_rigor?: InvestigationRigor;
  conclusion_validity?: ConclusionValidity;
  inquiry_completeness_permille: number;
  recommendations: string[];
}

/** Input for inquiry assessment */
export interface InquiryInput {
  question: string;
  hypothesis?: string;
  investigation_notes?: string;
  conclusions?: string;
  phases_completed: InquiryPhase[];
}

// =============================================================================
// Extended Adaptive Zome Functions (Features 15-21)
// =============================================================================

export interface AdaptiveZomeCriticalThinkingFunctions {
  // Feature 15: Critical Thinking
  analyze_argument: (input: ArgumentInput) => Promise<ArgumentAnalysis>;
  detect_fallacies: (text: string) => Promise<DetectedFallacy[]>;

  // Feature 16: Epistemic Vigilance
  evaluate_source: (input: SourceEvaluationInput) => Promise<SourceCredibility>;
  detect_biases: (input: BiasDetectionInput) => Promise<BiasAnalysis>;

  // Feature 17: Socratic Dialogue
  generate_socratic_questions: (input: SocraticInput) => Promise<SocraticDialogue[]>;
  steel_man_argument: (argument: string) => Promise<SteelMannedArgument>;

  // Feature 18: Metacognition
  assess_metacognition: (input: MetacognitiveInput) => Promise<MetacognitiveAssessment>;

  // Feature 19: Collaborative Knowledge Building
  assess_collaboration: (input: CollaborativeInput) => Promise<CollaborativeAssessment>;

  // Feature 20: Creativity & Divergent Thinking
  assess_creativity: (input: CreativityInput) => Promise<CreativityAssessment>;

  // Feature 21: Inquiry-Based Learning
  assess_inquiry: (input: InquiryInput) => Promise<InquiryAssessment>;
}

// =============================================================================
// TIER 1: Emotional & Affective Learning (Pekrun's Control-Value Theory)
// =============================================================================

/** Academic emotions based on Pekrun's Control-Value Theory */
export type AcademicEmotion =
  // Positive activating
  | 'Enjoyment'   // Engaged and enjoying learning
  | 'Hope'        // Expecting success
  | 'Pride'       // After achievement
  // Positive deactivating
  | 'Relief'      // After difficulty passed
  | 'Contentment' // Satisfied with progress
  // Negative activating
  | 'Anxiety'     // Fear of failure
  | 'Anger'       // Frustration with obstacles
  | 'Shame'       // After perceived failure
  // Negative deactivating
  | 'Hopelessness' // Giving up
  | 'Boredom';     // Disengaged

/** Emotional valence quadrants (valence x activation) */
export type EmotionalValence =
  | 'PositiveActivating'   // Joy, hope, pride
  | 'PositiveDeactivating' // Relief, contentment
  | 'NegativeActivating'   // Anxiety, anger, shame
  | 'NegativeDeactivating'; // Hopelessness, boredom

/** Emotional regulation strategies (Gross's Process Model + modern additions) */
export type EmotionalRegulationStrategy =
  // Gross's original 5
  | 'SituationSelection'     // Choose learning contexts
  | 'SituationModification'  // Modify difficulty/pace
  | 'AttentionalDeployment'  // Focus on specific aspects
  | 'CognitiveReappraisal'   // Reframe challenges positively
  | 'ExpressionSuppression'  // Control emotional display
  // Modern additions
  | 'AcceptanceAndCommitment' // ACT-based acceptance
  | 'MindfulnessAwareness'    // Present-moment awareness
  | 'SocialSupport'           // Seeking help from others
  | 'CollaborativeCoping';    // Group-based strategies

/** Frustration states with learning implications */
export type FrustrationState =
  | 'None'        // No frustration
  | 'Productive'  // Desirable difficulty (beneficial)
  | 'Mounting'    // Increasing but manageable
  | 'Peak'        // High frustration
  | 'Destructive' // Harmful to learning
  | 'Recovery';   // Cooling down after peak

/** Confusion states (D'Mello & Graesser research) */
export type ConfusionState =
  | 'NoConfusion'         // Clear understanding
  | 'ProductiveConfusion' // Optimal for deep learning!
  | 'Stuck'               // Needs intervention
  | 'Overwhelmed'         // Too much confusion
  | 'Disengaged';         // Given up on understanding

/** Current emotional state tracking */
export interface EmotionalState {
  primary_emotion: AcademicEmotion;
  intensity_permille: number;
  valence: EmotionalValence;
  frustration: FrustrationState;
  confusion: ConfusionState;
  emotional_stability_permille: number;
  duration_minutes: number;
}

/** Input for emotional assessment */
export interface EmotionAssessmentInput {
  learner: AgentPubKey;
  errors_last_5_minutes: number;
  help_requests: number;
  pause_frequency: number;
  response_time_variance_permille: number;
  self_reported_feeling?: AcademicEmotion;
  accuracy_trend: TrendDirection;
  session_duration_minutes: number;
}

/** Result of emotional assessment */
export interface EmotionAssessment {
  detected_state: EmotionalState;
  confidence_permille: number;
  regulation_strategies: EmotionalRegulationStrategy[];
  recommended_activities: string[];
  optimal_for_learning: boolean;
  intervention_urgency: number;
}

// =============================================================================
// TIER 2: Deliberate Practice Framework (Ericsson)
// =============================================================================

/** Phases of deliberate practice */
export type DeliberatePracticePhase =
  | 'SkillAnalysis'      // Identify components
  | 'StretchGoalSetting' // Set challenging targets
  | 'FocusedRepetition'  // Concentrated practice
  | 'ImmediateFeedback'  // Get quick feedback
  | 'Refinement'         // Adjust based on feedback
  | 'Consolidation';     // Lock in learning

/** Feedback timing impact on learning */
export type FeedbackTiming =
  | 'Immediate'     // Within seconds (best)
  | 'NearImmediate' // Within minutes
  | 'Delayed'       // Hours to days
  | 'VeryDelayed';  // Weeks (minimal benefit)

/** Feedback specificity levels */
export type FeedbackSpecificity =
  | 'Precise'   // Exact what/why/how
  | 'Targeted'  // Specific area identified
  | 'General'   // Overall direction
  | 'Vague';    // Unhelpful generalities

/** Stretch zone for optimal challenge */
export type StretchZone =
  | 'TooEasy'       // Comfort zone - no growth
  | 'OptimalStretch' // Just beyond ability (ideal)
  | 'ModerateStretch' // Challenging but doable
  | 'OverStretch';    // Too difficult - counterproductive

/** Mental representation quality (expert indicators) */
export interface MentalRepresentationQuality {
  pattern_recognition_permille: number;
  chunking_ability_permille: number;
  anticipation_accuracy_permille: number;
  error_detection_permille: number;
  self_monitoring_permille: number;
}

/** Skill decomposition for targeted practice */
export interface SkillDecomposition {
  parent_skill_id: string;
  sub_skills: string[];
  weakest_sub_skill: string;
  practice_priority_order: string[];
  estimated_hours_to_improve: number;
}

/** Deliberate practice quality assessment */
export interface DeliberatePracticeQuality {
  focus_intensity_permille: number;
  stretch_zone: StretchZone;
  feedback_timing: FeedbackTiming;
  feedback_specificity: FeedbackSpecificity;
  repetition_with_variation: boolean;
  goal_specificity_permille: number;
  overall_quality_permille: number;
}

// =============================================================================
// TIER 3: Social-Emotional Learning (CASEL Framework)
// =============================================================================

/** CASEL's 5 core SEL competencies */
export type SELCompetency =
  | 'SelfAwareness'
  | 'SelfManagement'
  | 'SocialAwareness'
  | 'RelationshipSkills'
  | 'ResponsibleDecisionMaking';

/** Self-awareness sub-skills */
export type SelfAwarenessSkill =
  | 'EmotionIdentification'
  | 'StrengthRecognition'
  | 'LimitationAwareness'
  | 'ConfidenceCalibration'
  | 'ValuesClarification'
  | 'GrowthMindsetOrientation';

/** Self-management sub-skills */
export type SelfManagementSkill =
  | 'ImpulseControl'
  | 'StressManagement'
  | 'SelfMotivation'
  | 'GoalSetting'
  | 'OrganizationalSkills'
  | 'SelfDiscipline';

/** Social awareness sub-skills */
export type SocialAwarenessSkill =
  | 'PerspectiveTaking'
  | 'Empathy'
  | 'DiversityAppreciation'
  | 'RespectForOthers'
  | 'SocialCueReading'
  | 'ContextualAwareness';

/** Relationship sub-skills */
export type RelationshipSkill =
  | 'Communication'
  | 'SocialEngagement'
  | 'CollaborativeTeamwork'
  | 'ConflictResolution'
  | 'SeekingHelp'
  | 'OfferingHelp';

/** Decision-making sub-skills */
export type DecisionMakingSkill =
  | 'ProblemIdentification'
  | 'AlternativeGeneration'
  | 'ConsequenceEvaluation'
  | 'EthicalReasoning'
  | 'ReflectiveAnalysis'
  | 'CriticalThinking';

/** SEL profile across all 5 competencies */
export interface SELProfile {
  self_awareness_permille: number;
  self_management_permille: number;
  social_awareness_permille: number;
  relationship_skills_permille: number;
  decision_making_permille: number;
  overall_sel_permille: number;
  strongest_competency: SELCompetency;
  growth_area: SELCompetency;
}

/** SEL intervention types */
export type SELIntervention =
  // Self-awareness interventions
  | 'JournalingPrompt'
  | 'StrengthsInventory'
  | 'FeelingVocabularyBuilding'
  // Self-management interventions
  | 'BreathingExercise'
  | 'GoalSettingWorksheet'
  | 'TimeManagementTool'
  // Social awareness interventions
  | 'PerspectiveTakingExercise'
  | 'CulturalExplorationActivity'
  | 'EmpathyMapping'
  // Relationship interventions
  | 'ActiveListeningPractice'
  | 'ConflictScenarioRolePlay'
  | 'CollaborativeProject'
  // Decision-making interventions
  | 'EthicalDilemmaDiscussion'
  | 'ConsequenceMapping'
  | 'ReflectionPrompt';

/** SEL assessment result */
export interface SELAssessment {
  profile: SELProfile;
  interventions: SELIntervention[];
  growth_trajectory: TrendDirection;
  peer_comparison_percentile: number;
  recommended_activities: string[];
}

// =============================================================================
// TIER 4: Stealth & Dynamic Assessment (Vygotsky ZPD)
// =============================================================================

/** Stealth assessment indicators (embedded in learning activities) */
export type StealthIndicator =
  // Time-based indicators
  | 'TimeToFirstAction'
  | 'PausePatterns'
  | 'ResponseTimeVariance'
  // Action-based indicators
  | 'ToolUsagePatterns'
  | 'HintRequestTiming'
  | 'RevisionBehavior'
  // Navigation indicators
  | 'PathThroughContent'
  | 'RevisitPatterns'
  | 'SkippingBehavior'
  // Social indicators
  | 'HelpSeekingPatterns'
  | 'PeerInteractionQuality'
  | 'ExplanationDepth';

/** Evidence types for stealth assessment */
export type AssessmentEvidence =
  | 'DirectPerformance'
  | 'ProcessEvidence'
  | 'TransferEvidence'
  | 'ExplanationEvidence'
  | 'PeerTeachingEvidence'
  | 'SelfAssessmentEvidence';

/** Vygotsky ZPD scaffolding levels */
export type ScaffoldingLevel =
  | 'NoSupport'   // Can do independently
  | 'Minimal'     // Light hints
  | 'Moderate'    // Structured guidance
  | 'Substantial' // Step-by-step support
  | 'Maximal';    // Full demonstration

/** Learning potential (ZPD width) */
export interface LearningPotential {
  current_performance_permille: number;
  supported_performance_permille: number;
  learning_potential_gap: number;
  scaffolding_responsiveness: number;
  transfer_potential_permille: number;
  modifiability_score_permille: number;
}

/** Self-assessment accuracy tracking */
export interface SelfAssessmentAccuracy {
  predicted_score_permille: number;
  actual_score_permille: number;
  accuracy_permille: number;
  bias_direction: number;
  calibration_trend: CalibrationTrend;
}

/** Stealth assessment data */
export interface StealthData {
  indicators: StealthIndicator[];
  evidence: AssessmentEvidence[];
  inferred_mastery_permille: number;
  confidence_permille: number;
}

/** Dynamic assessment result */
export interface DynamicAssessmentResult {
  learning_potential: LearningPotential;
  optimal_scaffolding: ScaffoldingLevel;
  responsiveness_to_support: number;
  recommended_intervention: string;
}

// =============================================================================
// TIER 5: Universal Design for Learning (CAST Framework)
// =============================================================================

/** UDL's three core principles */
export type UDLPrinciple =
  | 'Engagement'       // The "why" of learning
  | 'Representation'   // The "what" of learning
  | 'ActionExpression'; // The "how" of learning

/** Engagement options (9 checkpoints) */
export type EngagementOption =
  // Recruiting interest
  | 'ChoiceAndAutonomy'
  | 'RelevanceAuthenticity'
  | 'ThreatsDistractionsMin'
  // Sustaining effort
  | 'GoalsSaliency'
  | 'ChallengeSupport'
  | 'CollaborationCommunity'
  // Self-regulation
  | 'ExpectationsBeliefs'
  | 'CopingSkills'
  | 'SelfAssessmentReflection';

/** Representation options (9 checkpoints) */
export type RepresentationOption =
  // Perception
  | 'CustomizableDisplay'
  | 'AlternativesAuditory'
  | 'AlternativesVisual'
  // Language & symbols
  | 'VocabularySupport'
  | 'SymbolDecoding'
  | 'MultipleLanguages'
  // Comprehension
  | 'BackgroundKnowledge'
  | 'PatternsRelationships'
  | 'TransferGeneralization';

/** Action/Expression options (9 checkpoints) */
export type ActionExpressionOption =
  // Physical action
  | 'ResponseMethods'
  | 'NavigationAccess'
  | 'AssistiveTech'
  // Expression & communication
  | 'MultipleMedia'
  | 'ToolsComposition'
  | 'ScaffoldedPractice'
  // Executive function
  | 'GoalSettingSupport'
  | 'ProgressMonitoring'
  | 'CapacityManagement';

/** Accessibility needs categories */
export type AccessibilityNeed =
  | 'Visual'
  | 'Auditory'
  | 'Motor'
  | 'Cognitive'
  | 'Linguistic'
  | 'Emotional';

/** Learner UDL profile */
export interface UDLProfile {
  // Engagement preferences
  preferred_choice_level: number;
  collaboration_preference_permille: number;
  self_regulation_support_needed: boolean;
  // Representation preferences
  visual_preference_permille: number;
  auditory_preference_permille: number;
  text_preference_permille: number;
  needs_vocabulary_support: boolean;
  preferred_language: string;
  // Action preferences
  preferred_input_method: string;
  preferred_output_format: string;
  executive_function_support_needed: boolean;
  // Accessibility
  accessibility_needs: AccessibilityNeed[];
}

/** UDL barrier identification */
export interface UDLBarrier {
  barrier_type: UDLPrinciple;
  severity_permille: number;
  description: string;
  recommended_accommodations: string[];
}

/** UDL assessment result */
export interface UDLAssessment {
  profile: UDLProfile;
  barriers: UDLBarrier[];
  engagement_score_permille: number;
  representation_score_permille: number;
  action_expression_score_permille: number;
  recommended_accommodations: string[];
}

// =============================================================================
// Extended Adaptive Zome Functions (5 New Tiers)
// =============================================================================

export interface AdaptiveZomeEmotionalFunctions {
  // Tier 1: Emotional & Affective Learning
  assess_emotional_state: (input: EmotionAssessmentInput) => Promise<EmotionAssessment>;
  get_emotion_history: (learner: AgentPubKey) => Promise<EmotionalState[]>;
  recommend_regulation_strategy: (state: EmotionalState) => Promise<EmotionalRegulationStrategy[]>;

  // Tier 2: Deliberate Practice
  assess_deliberate_practice: (input: DeliberatePracticeQuality) => Promise<DeliberatePracticeQuality>;
  decompose_skill: (skill_hash: ActionHash) => Promise<SkillDecomposition>;
  get_mental_representation_quality: (skill_hash: ActionHash) => Promise<MentalRepresentationQuality>;

  // Tier 3: Social-Emotional Learning
  assess_sel_competencies: (learner: AgentPubKey) => Promise<SELAssessment>;
  recommend_sel_intervention: (profile: SELProfile) => Promise<SELIntervention[]>;

  // Tier 4: Stealth & Dynamic Assessment
  perform_stealth_assessment: (data: StealthData) => Promise<number>;
  perform_dynamic_assessment: (learner: AgentPubKey, skill_hash: ActionHash) => Promise<DynamicAssessmentResult>;

  // Tier 5: Universal Design for Learning
  assess_udl_needs: (profile: UDLProfile) => Promise<UDLAssessment>;
  recommend_accommodations: (barriers: UDLBarrier[]) => Promise<string[]>;
}

// =============================================================================
// TIER 6: GAMIFICATION 2.0 - Leaderboards, Achievements, Social
// Based on: Deterding (2011), Hamari (2014), Festinger (1954)
// =============================================================================

// Note: LeaderboardType is defined earlier (line ~421) for zome compatibility
// This is an extended version for Gamification 2.0 features
export type LeaderboardTypeExtended =
  | 'Global'      // All learners
  | 'Course'      // Within a course
  | 'Friends'     // Friend network only
  | 'Weekly'      // Weekly reset
  | 'Monthly'     // Monthly reset
  | 'AllTime';    // All-time records

// Note: LeaderboardEntry is defined earlier (line ~425) for zome compatibility
// This is an extended version for Gamification 2.0 features
export interface LeaderboardEntryExtended {
  rank: number;
  agent_id: string;
  display_name: string;
  score: number;
  xp_total: number;
  streak_days: number;
  badges_earned: number;
  mastery_skills: number;
}

export type AchievementCategory =
  | 'Learning'     // Complete lessons, master skills
  | 'Consistency'  // Streaks, daily goals
  | 'Social'       // Help others, collaborate
  | 'Mastery'      // Deep expertise
  | 'Exploration'  // Try new topics
  | 'Speed'        // Quick completion
  | 'Accuracy'     // High scores
  | 'Creative';    // Unique solutions

export type AchievementTier =
  | 'Bronze'      // 1x XP
  | 'Silver'      // 1.25x XP
  | 'Gold'        // 1.5x XP
  | 'Platinum'    // 2x XP
  | 'Diamond'     // 2.5x XP
  | 'Legendary';  // 3x XP

export interface Achievement {
  id: string;
  name: string;
  description: string;
  category: AchievementCategory;
  tier: AchievementTier;
  xp_reward: number;
  icon_url?: string;
  requirements: AchievementRequirement[];
}

export interface AchievementRequirement {
  metric: string;
  threshold: number;
  current_value: number;
}

export type ChallengeType =
  | 'Daily'       // 24-hour challenges
  | 'Weekly'      // 7-day challenges
  | 'Event'       // Special event challenges
  | 'Skill'       // Skill-specific challenges
  | 'Community'   // Group challenges
  | 'Versus';     // Head-to-head

export interface Challenge {
  id: string;
  challenge_type: ChallengeType;
  title: string;
  description: string;
  objectives: ChallengeObjective[];
  xp_reward: number;
  time_remaining_seconds: number;
  participants?: number;
}

export interface ChallengeObjective {
  description: string;
  target: number;
  current: number;
  completed: boolean;
}

// Festinger's Social Comparison Theory
export type SocialComparisonType =
  | 'UpwardClose'      // Compare to slightly better (aspiration)
  | 'UpwardFar'        // Compare to much better (inspiration)
  | 'Lateral'          // Compare to similar (validation)
  | 'DownwardClose'    // Compare to slightly worse (confidence)
  | 'SelfReferenced';  // Compare to past self (growth mindset)

export interface SocialComparison {
  comparison_type: SocialComparisonType;
  peer_display_name: string;
  peer_score: number;
  your_score: number;
  difference_permille: number;
  motivational_message: string;
}

export interface LeaderboardInput {
  leaderboard_type: LeaderboardType;
  course_hash?: ActionHash;
  limit?: number;
}

export interface AchievementCheckInput {
  learner: AgentPubKey;
  category?: AchievementCategory;
}

// =============================================================================
// TIER 7: ADVANCED ANALYTICS - Cohort Analysis, Predictive Models
// Based on: Baker & Inventado (2014), Romero & Ventura (2020)
// =============================================================================

export type CohortType =
  | 'EnrollmentDate'   // When they joined
  | 'SkillLevel'       // Beginner/Intermediate/Advanced
  | 'LearningStyle'    // VARK style
  | 'GoalType'         // What they want to achieve
  | 'ActivityPattern'  // How they engage
  | 'Custom';          // Custom cohort definition

export interface CohortStats {
  cohort_id: string;
  cohort_type: CohortType;
  member_count: number;
  avg_mastery_permille: number;
  avg_completion_rate_permille: number;
  avg_retention_rate_permille: number;
  avg_session_minutes: number;
  avg_days_to_mastery: number;
  top_performing_skills: string[];
  struggling_skills: string[];
}

export interface TrajectoryPoint {
  timestamp: number;
  mastery_permille: number;
  xp_total: number;
  skills_mastered: number;
  streak_days: number;
  engagement_permille: number;
}

export type TrajectoryType =
  | 'Steady'        // Consistent progress
  | 'FrontLoaded'   // Fast start, slowing
  | 'BackLoaded'    // Slow start, accelerating
  | 'Sporadic'      // Inconsistent
  | 'Plateaued'     // Stuck at level
  | 'Declining'     // Losing progress
  | 'Accelerating'; // Getting faster

// Note: TrendDirection is defined earlier (line ~1382) with values 'Up' | 'Down' | 'Stable'
// This is an alternative version for analytics (using different terminology)
export type TrendDirectionAnalytics = 'Increasing' | 'Decreasing' | 'Stable';

export interface LearningTrajectory {
  agent_id: string;
  trajectory_points: TrajectoryPoint[];
  trend: TrendDirectionAnalytics;
  velocity_permille: number;
  predicted_mastery_30_days: number;
  predicted_completion_date?: number;
  trajectory_type: TrajectoryType;
}

export type PredictionType =
  | 'Completion'       // Will they finish?
  | 'Dropout'          // Will they leave?
  | 'NextSkill'        // What should they learn next?
  | 'OptimalTime'      // When is best to study?
  | 'AssessmentScore'  // How will they score?
  | 'TimeToMastery';   // How long to master?

export interface PredictionFactor {
  factor_name: string;
  impact_permille: number;  // Can be negative
  current_value: string;
  optimal_value: string;
}

export interface Prediction {
  prediction_type: PredictionType;
  predicted_value: number;
  confidence_permille: number;
  factors: PredictionFactor[];
  recommendation: string;
}

export type AnomalyType =
  | 'PerformanceDrop'   // Sudden decline
  | 'UnusualTiming'     // Odd study times
  | 'SpeedAnomaly'      // Too fast/slow
  | 'EngagementShift'   // Behavior change
  | 'RoutineBreak'      // Pattern disruption
  | 'SkillRegression';  // Forgetting skills

export interface LearningAnomaly {
  anomaly_type: AnomalyType;
  severity_permille: number;
  detected_at: number;
  description: string;
  recommended_action: string;
}

export interface AtRiskIndicator {
  risk_type: string;
  risk_score_permille: number;
  contributing_factors: string[];
  early_warning_days: number;
  intervention_suggestions: string[];
}

export interface CohortAnalysisInput {
  cohort_type: CohortType;
  cohort_filter?: string;
  comparison_cohort?: string;
}

export interface TrajectoryInput {
  agent_id: string;
  days_to_analyze?: number;
}

export interface PredictionInput {
  agent_id: string;
  prediction_type: PredictionType;
  skill_hash?: ActionHash;
  goal_hash?: ActionHash;
}

// =============================================================================
// TIER 8: AI TUTORING INTEGRATION - Hints, Explanations, Conversation
// Based on: VanLehn (2011), Graesser et al. (2004)
// =============================================================================

export type TutoringMode =
  | 'Socratic'         // Deep questioning
  | 'Direct'           // Explicit instruction
  | 'GuidedDiscovery'  // Lead to discovery
  | 'WorkedExamples'   // Show step-by-step
  | 'Scaffolded'       // Gradually reduce support
  | 'Remediation';     // Fix misunderstandings

export type HintLevel =
  | 'Metacognitive'  // "Think about what you already know" (0% XP penalty)
  | 'Directional'    // "Focus on the denominator" (10% XP penalty)
  | 'Specific'       // "The denominator should be 12" (25% XP penalty)
  | 'BottomOut';     // Full answer revealed (50% XP penalty)

export interface HintResponse {
  hint_level: HintLevel;
  hint_text: string;
  remaining_hints: number;
  xp_penalty_if_used: number;
  related_concepts: string[];
  try_first_suggestions: string[];
}

export type ExplanationType =
  | 'Conceptual'    // What and why
  | 'Procedural'    // Step-by-step how
  | 'ExampleBased'  // Learn from examples
  | 'Analogy'       // Compare to familiar
  | 'Visual'        // Diagrams and images
  | 'Comparative';  // Compare/contrast

export interface PersonalizedExplanation {
  explanation_type: ExplanationType;
  content: string;
  complexity_level: number;  // Matched to learner
  examples: string[];
  analogies: string[];
  key_points: string[];
  common_misconceptions: string[];
  follow_up_questions: string[];
}

export type FeedbackType =
  | 'Correct'          // Right answer
  | 'PartiallyCorrect' // Partially right
  | 'Incorrect'        // Wrong answer
  | 'Effort'           // Recognize effort
  | 'Progress'         // Show improvement
  | 'Mastery';         // Celebrate mastery

export interface PersonalizedFeedback {
  feedback_type: FeedbackType;
  message: string;
  specific_praise?: string;
  growth_mindset_message?: string;
  next_steps: string[];
  emotional_tone: string;  // "encouraging", "celebratory", "supportive"
}

export interface DialogueTurn {
  speaker: 'tutor' | 'learner';
  message: string;
  intent: string;  // "question", "answer", "hint", "explanation"
  timestamp: number;
  confidence_permille?: number;
}

export interface TutoringSession {
  session_id: string;
  skill_hash: ActionHash;
  tutoring_mode: TutoringMode;
  dialogue_history: DialogueTurn[];
  hints_used: number;
  explanations_requested: number;
  current_problem_attempts: number;
  session_mastery_start: number;
  session_mastery_current: number;
  emotional_state_detected: string;
}

export interface DetectedMisconception {
  misconception_id: string;
  description: string;
  evidence: string[];
  correct_understanding: string;
  remediation_approach: string;
  common_sources: string[];
}

export interface TutoringStrategy {
  recommended_mode: TutoringMode;
  reasoning: string;
  estimated_time_minutes: number;
  key_concepts_to_cover: string[];
  potential_obstacles: string[];
  adaptive_triggers: string[];
}

export interface HintRequestInput {
  skill_hash: ActionHash;
  problem_id: string;
  current_attempt: string;
  hints_already_used: number;
}

export interface ExplanationRequestInput {
  skill_hash: ActionHash;
  concept: string;
  preferred_type?: ExplanationType;
  learner_mastery: number;
}

export interface FeedbackInput {
  skill_hash: ActionHash;
  problem_id: string;
  learner_answer: string;
  correct_answer: string;
  attempt_number: number;
}

export interface MisconceptionInput {
  skill_hash: ActionHash;
  recent_errors: string[];
  answer_patterns: string[];
}

export interface TutoringStrategyInput {
  skill_hash: ActionHash;
  learner_mastery: number;
  learning_style: LearningStyle;
  recent_errors: number;
  time_available_minutes: number;
  goal_type: GoalType;
}

// =============================================================================
// Extended Adaptive Zome Functions (Tiers 6-8)
// =============================================================================

export interface AdaptiveZomeGamification2Functions {
  // Tier 6: Gamification 2.0
  get_leaderboard: (input: LeaderboardInput) => Promise<LeaderboardEntry[]>;
  check_achievements: (input: AchievementCheckInput) => Promise<Achievement[]>;
  get_social_comparison: (learner: AgentPubKey) => Promise<SocialComparison>;
  get_active_challenges: (learner: AgentPubKey) => Promise<Challenge[]>;
}

export interface AdaptiveZomeAnalyticsFunctions {
  // Tier 7: Advanced Analytics
  analyze_cohort: (input: CohortAnalysisInput) => Promise<CohortStats>;
  analyze_trajectory: (input: TrajectoryInput) => Promise<LearningTrajectory>;
  generate_prediction: (input: PredictionInput) => Promise<Prediction>;
  detect_anomalies: (learner: AgentPubKey) => Promise<LearningAnomaly[]>;
  assess_at_risk: (learner: AgentPubKey) => Promise<AtRiskIndicator>;
}

export interface AdaptiveZomeTutoringFunctions {
  // Tier 8: AI Tutoring Integration
  request_hint: (input: HintRequestInput) => Promise<HintResponse>;
  request_explanation: (input: ExplanationRequestInput) => Promise<PersonalizedExplanation>;
  generate_feedback: (input: FeedbackInput) => Promise<PersonalizedFeedback>;
  detect_misconceptions: (input: MisconceptionInput) => Promise<DetectedMisconception[]>;
  get_tutoring_strategy: (input: TutoringStrategyInput) => Promise<TutoringStrategy>;
}

// =============================================================================
// Combined Zome Interface
// =============================================================================

export interface ZomeFunctions {
  learning: LearningZomeFunctions;
  fl: FlZomeFunctions;
  credential: CredentialZomeFunctions;
  dao: DaoZomeFunctions;
  srs: SrsZomeFunctions;
  gamification: GamificationZomeFunctions;
  adaptive: AdaptiveZomeFunctions
    & AdaptiveZomeCriticalThinkingFunctions
    & AdaptiveZomeEmotionalFunctions
    & AdaptiveZomeGamification2Functions
    & AdaptiveZomeAnalyticsFunctions
    & AdaptiveZomeTutoringFunctions;
  integration: IntegrationZomeFunctions;
}
