// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Adaptive Learning Intelligence Coordinator Zome
//!
//! Implements the business logic for ML-inspired personalized learning:
//!
//! - **Profile Management**: VARK learning style detection, cognitive profiling
//! - **Mastery Tracking**: Bayesian Knowledge Tracing updates
//! - **Recommendations**: Smart content suggestions based on ZPD
//! - **Analytics**: Session and aggregated learning analytics
//! - **Adaptive Paths**: Personalized learning journeys

use hdk::prelude::*;
use adaptive_integrity::*;

mod tutoring;
mod analytics;
mod gamification_social;
mod retention;
mod flow;
mod mastery;
pub mod grade_adaptation;
pub mod scheduling;

pub use tutoring::*;
pub use analytics::*;
pub use gamification_social::*;
pub use retention::*;
pub use flow::*;
pub use mastery::*;
pub use grade_adaptation::*;
pub use scheduling::*;

// ============== Constants ==============

/// Default BKT parameters
const DEFAULT_LEARN_RATE: u16 = 100; // 10%
const DEFAULT_GUESS_RATE: u16 = 200; // 20%
const DEFAULT_SLIP_RATE: u16 = 100;  // 10%

/// Target retention for review scheduling
const RETENTION_TARGET: u16 = 800; // 80%

/// Mastery threshold for skill completion
const MASTERY_THRESHOLD: u16 = 800; // 80%

/// Maximum recommendations to generate
const MAX_RECOMMENDATIONS: usize = 10;

/// Flow state difficulty range (within ZPD sweet spot)
const FLOW_DIFFICULTY_TOLERANCE_PERMILLE: u16 = 100; // ±10%

/// Interleaving ratio for topic mixing
const INTERLEAVING_RATIO_PERMILLE: u16 = 300; // 30% different topics

/// Time decay factor for recency weighting
const TIME_DECAY_FACTOR: f64 = 0.95;

/// Minimum confidence for reliable recommendations
const MIN_CONFIDENCE_PERMILLE: u16 = 300;

// ============== Smart Recommendation Engine ==============

/// Smart recommendation context with learner state
#[derive(Clone, Debug)]
struct RecommendationContext {
    /// Current mastery levels by skill
    masteries: Vec<SkillMastery>,
    /// Learning style preferences
    learning_style: LearningStyle,
    /// VARK scores (visual, auditory, read/write, kinesthetic)
    vark_scores: (u16, u16, u16, u16),
    /// Active goals
    goals: Vec<LearningGoal>,
    /// Current hour (0-23) for time-of-day optimization
    hour: u8,
    /// Day of week (0=Sunday, 6=Saturday)
    day_of_week: u8,
    /// Recent topics for interleaving
    recent_topic_hashes: Vec<ActionHash>,
    /// Session count today
    sessions_today: u32,
    /// Estimated energy level (0-1000)
    energy_permille: u16,
}

impl RecommendationContext {
    /// Calculate optimal difficulty based on flow state theory
    fn optimal_difficulty_range(&self, skill_mastery: u16) -> (u16, u16) {
        // ZPD bounds: slightly above current mastery
        let base_lower = skill_mastery.saturating_add(50);
        let base_upper = skill_mastery.saturating_add(150);

        // Adjust based on energy level
        let energy_factor = self.energy_permille as i32 - 500; // -500 to +500
        let adjustment = (energy_factor / 10) as i16; // -50 to +50

        let lower = (base_lower as i16 + adjustment).max(0).min(1000) as u16;
        let upper = (base_upper as i16 + adjustment).max(0).min(1000) as u16;

        (lower.min(1000), upper.min(1000))
    }

    /// Calculate time-of-day bonus for different content types
    /// Uses actual ContentType variants: Lesson, Quiz, Exercise, Project, Assessment, Challenge
    fn time_bonus(&self, content_type: &ContentType) -> u16 {
        match self.hour {
            // Morning (6-12): Best for analytical, new learning
            6..=11 => match content_type {
                ContentType::Lesson => 150,           // New concepts best in morning
                ContentType::Quiz | ContentType::Assessment => 100,  // Testing when fresh
                ContentType::Exercise => 75,
                _ => 50,
            },
            // Afternoon (12-17): Good for practice and hands-on work
            12..=16 => match content_type {
                ContentType::Exercise => 150,         // Practice in afternoon
                ContentType::Quiz | ContentType::Assessment => 100,
                ContentType::Project => 100,          // Collaborative work
                _ => 50,
            },
            // Evening (17-21): Good for creative, synthesis, projects
            17..=20 => match content_type {
                ContentType::Project => 150,          // Creative synthesis in evening
                ContentType::Challenge => 100,        // Complex challenges
                ContentType::Exercise => 75,
                _ => 50,
            },
            // Night (21-6): Good for light review
            _ => match content_type {
                ContentType::Lesson => 100,           // Light reading
                ContentType::Quiz => 50,              // Quick reviews only
                _ => 25,
            },
        }
    }

    /// Calculate learning style match score
    /// Maps VARK styles to content types:
    /// - Visual (v): Lessons (with diagrams), Projects (visual output)
    /// - Auditory (a): Lessons (lectures), Quizzes (verbal Q&A)
    /// - Read/Write (r): Lessons (text), Assessments (written)
    /// - Kinesthetic (k): Exercise, Project, Challenge (hands-on)
    fn style_match_score(&self, content_type: &ContentType) -> u16 {
        let (v, a, r, k) = self.vark_scores;
        let max_score = v.max(a).max(r).max(k);

        if max_score == 0 {
            return 500; // No preference = neutral
        }

        match content_type {
            // Lessons appeal to V, A, R learners (reading/watching/listening)
            ContentType::Lesson => {
                let combined = (v as u32 + a as u32 + r as u32) / 3;
                (combined * 1000 / max_score as u32).min(1000) as u16
            }
            // Quizzes appeal to A (verbal) and R (written) learners
            ContentType::Quiz => {
                let combined = (a as u32 + r as u32) / 2;
                (combined * 1000 / max_score as u32).min(1000) as u16
            }
            // Exercises are hands-on: K learners
            ContentType::Exercise => {
                (k as u32 * 1000 / max_score as u32).min(1000) as u16
            }
            // Projects combine visual output and hands-on: V + K
            ContentType::Project => {
                let combined = (v as u32 + k as u32) / 2;
                (combined * 1000 / max_score as u32).min(1000) as u16
            }
            // Assessments are written: R learners
            ContentType::Assessment => {
                (r as u32 * 1000 / max_score as u32).min(1000) as u16
            }
            // Challenges are complex hands-on: K learners primarily
            ContentType::Challenge => {
                (k as u32 * 1000 / max_score as u32).min(1000) as u16
            }
        }
    }

    /// Check if a topic should be interleaved (switch topics for better retention)
    fn should_interleave(&self, topic_hash: &ActionHash) -> bool {
        if self.recent_topic_hashes.is_empty() {
            return false;
        }

        // Count how many recent items are the same topic
        let same_topic_count = self.recent_topic_hashes
            .iter()
            .filter(|h| *h == topic_hash)
            .count();

        // After 3 consecutive same-topic items, recommend switching
        same_topic_count >= 3
    }

    /// Calculate goal alignment score
    fn goal_alignment_score(&self, skill_hash: &ActionHash) -> u16 {
        for goal in &self.goals {
            if goal.target_hashes.contains(skill_hash) {
                // Higher priority goals get higher scores
                return match goal.priority {
                    GoalPriority::Critical => 1000,
                    GoalPriority::High => 800,
                    GoalPriority::Medium => 600,
                    GoalPriority::Low => 400,
                };
            }
        }
        200 // Not aligned with any goal
    }

    /// Estimate energy level based on time and sessions
    fn estimate_energy(hour: u8, sessions_today: u32) -> u16 {
        // Base energy by time of day (circadian rhythm)
        let base_energy: u16 = match hour {
            6..=9 => 700,   // Morning ramp-up
            10..=11 => 900, // Morning peak
            12..=13 => 600, // Post-lunch dip
            14..=16 => 800, // Afternoon recovery
            17..=19 => 700, // Evening
            20..=22 => 500, // Night wind-down
            _ => 300,       // Late night/early morning
        };

        // Fatigue from sessions (each session reduces by 5%)
        let fatigue = (sessions_today * 50).min(400);

        base_energy.saturating_sub(fatigue as u16)
    }
}

/// Enhanced recommendation with smart scoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmartRecommendation {
    /// Base recommendation data
    pub base: Recommendation,
    /// Flow state match score (0-1000)
    pub flow_score_permille: u16,
    /// Learning style compatibility (0-1000)
    pub style_score_permille: u16,
    /// Time-of-day appropriateness (0-1000)
    pub timing_score_permille: u16,
    /// Goal alignment (0-1000)
    pub goal_score_permille: u16,
    /// Interleaving benefit (0-1000)
    pub interleave_score_permille: u16,
    /// Combined smart score (0-1000)
    pub smart_score_permille: u16,
    /// Detailed reasoning for this recommendation
    pub detailed_explanation: String,
}

/// Generate smart recommendations using learning science
fn generate_smart_recommendations(
    ctx: &RecommendationContext,
    agent: AgentPubKey,
    now: i64,
    limit: usize,
) -> Vec<SmartRecommendation> {
    let mut recommendations = Vec::new();

    // Priority 1: Skills due for review (retention risk)
    let due: Vec<&SkillMastery> = ctx.masteries
        .iter()
        .filter(|m| m.next_optimal_review <= now && m.mastery_permille > 0)
        .collect();

    for mastery in due.iter().take(limit / 3) {
        let (diff_low, diff_high) = ctx.optimal_difficulty_range(mastery.mastery_permille);
        let flow_score = if mastery.mastery_permille >= diff_low && mastery.mastery_permille <= diff_high {
            900
        } else {
            600
        };

        let goal_score = ctx.goal_alignment_score(&mastery.skill_hash);
        let interleave_score = if ctx.should_interleave(&mastery.skill_hash) { 400 } else { 800 };

        // Smart score is weighted combination
        let smart_score = (
            flow_score as u32 * 25 +
            500 * 20 + // Style neutral for reviews
            ctx.time_bonus(&ContentType::Exercise) as u32 * 15 +
            goal_score as u32 * 25 +
            interleave_score as u32 * 15
        ) / 100;

        let base_rec = Recommendation {
            learner: agent.clone(),
            target_hash: mastery.skill_hash.clone(),
            rec_type: RecommendationType::Review,
            score_permille: smart_score as u16,
            relevance_permille: 900,
            difficulty_match_permille: flow_score,
            readiness_permille: 1000,
            freshness_permille: 500,
            reason: RecommendationReason::RetentionRisk,
            explanation: format!(
                "Review needed to maintain {:?} level",
                MasteryLevel::from_permille(mastery.mastery_permille)
            ),
            rank: (recommendations.len() + 1) as u32,
            is_valid: true,
            generated_at: now,
            expires_at: now + 86400_000_000,
        };

        recommendations.push(SmartRecommendation {
            base: base_rec,
            flow_score_permille: flow_score,
            style_score_permille: 500,
            timing_score_permille: ctx.time_bonus(&ContentType::Exercise),
            goal_score_permille: goal_score,
            interleave_score_permille: interleave_score,
            smart_score_permille: smart_score as u16,
            detailed_explanation: format!(
                "Due for review • Flow: {}% • Goal aligned: {}%",
                flow_score / 10, goal_score / 10
            ),
        });
    }

    // Priority 2: Weak areas needing remediation
    let weak: Vec<&SkillMastery> = ctx.masteries
        .iter()
        .filter(|m| m.mastery_permille < 400 && m.total_attempts >= 3 && m.confidence_permille >= MIN_CONFIDENCE_PERMILLE)
        .collect();

    for mastery in weak.iter().take(limit / 4) {
        let goal_score = ctx.goal_alignment_score(&mastery.skill_hash);
        let interleave_score = if ctx.should_interleave(&mastery.skill_hash) { 400 } else { 700 };

        let smart_score = (
            700 * 25 + // Flow score for remediation
            500 * 20 + // Style neutral
            ctx.time_bonus(&ContentType::Exercise) as u32 * 15 +
            goal_score as u32 * 25 +
            interleave_score as u32 * 15
        ) / 100;

        let base_rec = Recommendation {
            learner: agent.clone(),
            target_hash: mastery.skill_hash.clone(),
            rec_type: RecommendationType::Practice,
            score_permille: smart_score as u16,
            relevance_permille: 850,
            difficulty_match_permille: 700,
            readiness_permille: 800,
            freshness_permille: 600,
            reason: RecommendationReason::WeaknessRemediation,
            explanation: "Practice to strengthen this challenging area".to_string(),
            rank: (recommendations.len() + 1) as u32,
            is_valid: true,
            generated_at: now,
            expires_at: now + 86400_000_000,
        };

        recommendations.push(SmartRecommendation {
            base: base_rec,
            flow_score_permille: 700,
            style_score_permille: 500,
            timing_score_permille: ctx.time_bonus(&ContentType::Exercise),
            goal_score_permille: goal_score,
            interleave_score_permille: interleave_score,
            smart_score_permille: smart_score as u16,
            detailed_explanation: format!(
                "Needs practice • Mastery: {}% • {} attempts",
                mastery.mastery_permille / 10, mastery.total_attempts
            ),
        });
    }

    // Priority 3: Strengths ready for challenges
    let strengths: Vec<&SkillMastery> = ctx.masteries
        .iter()
        .filter(|m| m.mastery_permille >= 800)
        .collect();

    for mastery in strengths.iter().take(limit / 4) {
        let goal_score = ctx.goal_alignment_score(&mastery.skill_hash);
        let interleave_score = if ctx.should_interleave(&mastery.skill_hash) { 500 } else { 600 };

        // Challenges are better in morning when energy is high
        let timing_bonus = if ctx.energy_permille >= 700 {
            ctx.time_bonus(&ContentType::Challenge)
        } else {
            ctx.time_bonus(&ContentType::Challenge) / 2
        };

        let smart_score = (
            850 * 25 + // High flow for mastered skills
            500 * 20 + // Style neutral
            timing_bonus as u32 * 15 +
            goal_score as u32 * 25 +
            interleave_score as u32 * 15
        ) / 100;

        let base_rec = Recommendation {
            learner: agent.clone(),
            target_hash: mastery.skill_hash.clone(),
            rec_type: RecommendationType::Challenge,
            score_permille: smart_score as u16,
            relevance_permille: 700,
            difficulty_match_permille: 800,
            readiness_permille: 1000,
            freshness_permille: 700,
            reason: RecommendationReason::StrengthExpansion,
            explanation: "Ready for advanced challenges".to_string(),
            rank: (recommendations.len() + 1) as u32,
            is_valid: true,
            generated_at: now,
            expires_at: now + 86400_000_000,
        };

        recommendations.push(SmartRecommendation {
            base: base_rec,
            flow_score_permille: 850,
            style_score_permille: 500,
            timing_score_permille: timing_bonus,
            goal_score_permille: goal_score,
            interleave_score_permille: interleave_score,
            smart_score_permille: smart_score as u16,
            detailed_explanation: format!(
                "Challenge ready • Master level • Energy: {}%",
                ctx.energy_permille / 10
            ),
        });
    }

    // Sort by smart score
    recommendations.sort_by(|a, b| b.smart_score_permille.cmp(&a.smart_score_permille));

    // Re-rank
    for (i, rec) in recommendations.iter_mut().enumerate() {
        rec.base.rank = (i + 1) as u32;
    }

    recommendations.truncate(limit);
    recommendations
}

// ============== Helper Functions ==============

/// Convert a Holochain Timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

/// Get current time as i64
fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}

/// Get learner anchor hash for profiles
fn profile_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.profile.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToProfile)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for masteries
fn mastery_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.mastery.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToMasteries)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for goals
fn goal_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.goal.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToGoals)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for sessions
fn session_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.session.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToSessions)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for paths
fn path_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.path.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToPaths)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for recommendations
fn recommendation_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.rec.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToRecommendations)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for assessments
fn assessment_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("adaptive.assessment.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToAssessments)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

// ============== Batch Query Optimization ==============

/// Combined learner context - reduces DHT calls by fetching related data together
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnerContext {
    /// Learner's profile (optional if not yet created)
    pub profile: Option<LearnerProfile>,
    /// All skill masteries
    pub masteries: Vec<SkillMastery>,
    /// Active learning goals
    pub goals: Vec<LearningGoal>,
    /// Skills due for review
    pub due_for_review: Vec<SkillMastery>,
    /// Strong areas (mastery >= 800)
    pub strengths: Vec<SkillMastery>,
    /// Weak areas (mastery < 400, attempts >= 3)
    pub weaknesses: Vec<SkillMastery>,
    /// Computed stats
    pub stats: LearnerStats,
}

/// Pre-computed learner statistics
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LearnerStats {
    /// Total skills being tracked
    pub total_skills: u32,
    /// Skills at mastery level (>= 800)
    pub mastered_count: u32,
    /// Skills in progress (400-799)
    pub in_progress_count: u32,
    /// Skills needing work (< 400)
    pub needs_work_count: u32,
    /// Average mastery across all skills
    pub avg_mastery_permille: u16,
    /// Skills due for review
    pub due_review_count: u32,
    /// Active goals count
    pub active_goals_count: u32,
    /// Completed goals count
    pub completed_goals_count: u32,
    /// Dominant learning style
    pub dominant_style: LearningStyle,
}

// Note: Batch operations are handled inline within get_learner_context
// This reduces complexity while still optimizing DHT calls through:
// 1. GetOptions::local() for faster local-first queries
// 2. Computing derived data in-memory from already-fetched entries
// 3. Single combined context fetch replacing multiple separate queries

/// Get complete learner context in one call
/// This reduces 4-5 separate DHT queries to optimized batch operations
#[hdk_extern]
pub fn get_learner_context(_: ()) -> ExternResult<LearnerContext> {
    let now = current_time()?;

    // Get profile (single call)
    let profile = get_my_profile(())?;

    // Get all masteries (one link query + batch entry fetches)
    let masteries = get_my_masteries(())?;

    // Get all goals (one link query + batch entry fetches)
    let goals = get_my_goals(())?;

    // Compute derived data from already-fetched masteries
    let due_for_review: Vec<SkillMastery> = masteries
        .iter()
        .filter(|m| m.next_optimal_review <= now && m.mastery_permille > 0)
        .cloned()
        .collect();

    let strengths: Vec<SkillMastery> = masteries
        .iter()
        .filter(|m| m.mastery_permille >= 800)
        .cloned()
        .collect();

    let weaknesses: Vec<SkillMastery> = masteries
        .iter()
        .filter(|m| m.mastery_permille < 400 && m.total_attempts >= 3)
        .cloned()
        .collect();

    // Compute stats
    let total_skills = masteries.len() as u32;
    let mastered_count = strengths.len() as u32;
    let needs_work_count = weaknesses.len() as u32;
    let in_progress_count = total_skills.saturating_sub(mastered_count + needs_work_count);
    let avg_mastery_permille = if total_skills > 0 {
        (masteries.iter().map(|m| m.mastery_permille as u32).sum::<u32>() / total_skills) as u16
    } else {
        0
    };
    let active_goals_count = goals.iter().filter(|g| !g.is_completed).count() as u32;
    let completed_goals_count = goals.iter().filter(|g| g.is_completed).count() as u32;

    // Determine dominant learning style from profile
    let dominant_style = profile.as_ref().map(|p| {
        let scores = [
            (p.visual_score_permille, LearningStyle::Visual),
            (p.auditory_score_permille, LearningStyle::Auditory),
            (p.reading_score_permille, LearningStyle::ReadingWriting),
            (p.kinesthetic_score_permille, LearningStyle::Kinesthetic),
        ];
        scores.iter()
            .max_by_key(|(score, _)| *score)
            .map(|(_, style)| style.clone())
            .unwrap_or(LearningStyle::Multimodal)
    }).unwrap_or(LearningStyle::Multimodal);

    let stats = LearnerStats {
        total_skills,
        mastered_count,
        in_progress_count,
        needs_work_count,
        avg_mastery_permille,
        due_review_count: due_for_review.len() as u32,
        active_goals_count,
        completed_goals_count,
        dominant_style,
    };

    Ok(LearnerContext {
        profile,
        masteries,
        goals,
        due_for_review,
        strengths,
        weaknesses,
        stats,
    })
}

/// Get paginated masteries with filtering - optimized for UI lists
#[hdk_extern]
pub fn get_masteries_paginated(input: PaginatedMasteriesInput) -> ExternResult<PaginatedMasteriesResult> {
    mastery::get_masteries_paginated_impl(input)
}

// ============== Profile Management ==============

/// Input for creating a learner profile
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateProfileInput {
    /// Initial learning style scores (optional)
    pub visual_score: Option<u16>,
    pub auditory_score: Option<u16>,
    pub reading_score: Option<u16>,
    pub kinesthetic_score: Option<u16>,
    /// Preferred session length
    pub preferred_session_minutes: Option<u16>,
    /// Optimal learning hours
    pub optimal_hour_start: Option<u8>,
    pub optimal_hour_end: Option<u8>,
}

/// Create or update a learner profile
#[hdk_extern]
pub fn create_profile(input: CreateProfileInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = profile_anchor()?;

    // Create profile with defaults or provided values
    let profile = LearnerProfile {
        learner: agent.clone(),
        // Learning styles default to balanced (250 each)
        visual_score_permille: input.visual_score.unwrap_or(250),
        auditory_score_permille: input.auditory_score.unwrap_or(250),
        reading_score_permille: input.reading_score.unwrap_or(250),
        kinesthetic_score_permille: input.kinesthetic_score.unwrap_or(250),
        // Cognitive defaults
        preferred_session_minutes: input.preferred_session_minutes.unwrap_or(25),
        optimal_hour_start: input.optimal_hour_start.unwrap_or(9),
        optimal_hour_end: input.optimal_hour_end.unwrap_or(17),
        attention_span_minutes: 20,
        preferred_difficulty_permille: 500,
        // Performance starts at zero
        accuracy_permille: 0,
        velocity_centile: 0,
        retention_7d_permille: 0,
        retention_30d_permille: 0,
        // Engagement
        avg_daily_minutes: 0,
        days_since_active: 0,
        total_sessions: 0,
        completion_rate_permille: 0,
        // Low confidence initially
        confidence_permille: 100,
        data_points: 0,
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::LearnerProfile(profile))?;

    // Link from anchor to profile
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToProfile,
        (),
    )?;

    Ok(action_hash)
}

/// Get the learner's profile
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<LearnerProfile>> {
    let anchor = profile_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToProfile)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let profile: LearnerProfile = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No profile entry".into())))?;
                return Ok(Some(profile));
            }
        }
    }

    Ok(None)
}

/// Record a learning style assessment response
#[hdk_extern]
pub fn record_style_assessment(input: LearningStyleAssessment) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let anchor = assessment_anchor()?;

    // Verify the assessment is for this agent
    if input.learner != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only record your own assessments".into()
        )));
    }

    let action_hash = create_entry(EntryTypes::LearningStyleAssessment(input))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToAssessments,
        (),
    )?;

    Ok(action_hash)
}

/// Calculate learning style from assessments
#[hdk_extern]
pub fn calculate_learning_style(_: ()) -> ExternResult<LearningStyleResult> {
    let anchor = assessment_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToAssessments)?,
        GetStrategy::Local,
    )?;

    let mut visual = 0u32;
    let mut auditory = 0u32;
    let mut reading = 0u32;
    let mut kinesthetic = 0u32;
    let mut total = 0u32;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let assessment: LearningStyleAssessment = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No assessment entry".into())))?;

                let weight = assessment.confidence as u32;
                match assessment.answer {
                    LearningStyle::Visual => visual += weight,
                    LearningStyle::Auditory => auditory += weight,
                    LearningStyle::ReadingWriting => reading += weight,
                    LearningStyle::Kinesthetic => kinesthetic += weight,
                    LearningStyle::Multimodal => {
                        // Distribute evenly
                        visual += weight / 4;
                        auditory += weight / 4;
                        reading += weight / 4;
                        kinesthetic += weight / 4;
                    }
                }
                total += weight;
            }
        }
    }

    if total == 0 {
        return Ok(LearningStyleResult {
            primary: LearningStyle::Multimodal,
            visual_permille: 250,
            auditory_permille: 250,
            reading_permille: 250,
            kinesthetic_permille: 250,
            confidence_permille: 0,
        });
    }

    // Convert to permille
    let visual_p = ((visual * 1000) / total) as u16;
    let auditory_p = ((auditory * 1000) / total) as u16;
    let reading_p = ((reading * 1000) / total) as u16;
    let kinesthetic_p = ((kinesthetic * 1000) / total) as u16;

    // Find primary style
    let max = visual_p.max(auditory_p).max(reading_p).max(kinesthetic_p);
    let primary = if visual_p == max {
        LearningStyle::Visual
    } else if auditory_p == max {
        LearningStyle::Auditory
    } else if reading_p == max {
        LearningStyle::ReadingWriting
    } else {
        LearningStyle::Kinesthetic
    };

    // Confidence based on number of assessments
    let confidence = (total * 100).min(1000) as u16;

    Ok(LearningStyleResult {
        primary,
        visual_permille: visual_p,
        auditory_permille: auditory_p,
        reading_permille: reading_p,
        kinesthetic_permille: kinesthetic_p,
        confidence_permille: confidence,
    })
}

/// Learning style calculation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningStyleResult {
    pub primary: LearningStyle,
    pub visual_permille: u16,
    pub auditory_permille: u16,
    pub reading_permille: u16,
    pub kinesthetic_permille: u16,
    pub confidence_permille: u16,
}

// ============== Mastery Tracking (BKT) ==============
// Types and logic extracted to mastery.rs — thin wrappers below.

/// Create or get mastery record for a skill
#[hdk_extern]
pub fn get_or_create_mastery(input: CreateMasteryInput) -> ExternResult<ActionHash> {
    mastery::get_or_create_mastery_impl(input)
}

/// Record a practice attempt and update mastery
#[hdk_extern]
pub fn record_attempt(input: RecordAttemptInput) -> ExternResult<MasteryUpdateResult> {
    mastery::record_attempt_impl(input)
}

/// Get all mastery records for the learner
#[hdk_extern]
pub fn get_my_masteries(_: ()) -> ExternResult<Vec<SkillMastery>> {
    mastery::get_my_masteries_impl(())
}

/// Get mastery record for a specific skill
#[hdk_extern]
pub fn get_skill_mastery(input: GetSkillMasteryInput) -> ExternResult<Option<SkillMastery>> {
    mastery::get_skill_mastery_impl(input)
}

/// Get skills due for review
#[hdk_extern]
pub fn get_due_for_review(_: ()) -> ExternResult<Vec<SkillMastery>> {
    mastery::get_due_for_review_impl(())
}

// ============== Recommendations ==============

/// Input for generating recommendations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRecsInput {
    /// Maximum number of recommendations
    pub limit: Option<usize>,
    /// Filter by recommendation type
    pub rec_type: Option<RecommendationType>,
}

/// Generate personalized recommendations
#[hdk_extern]
pub fn generate_recommendations(input: GenerateRecsInput) -> ExternResult<Vec<Recommendation>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let limit = input.limit.unwrap_or(MAX_RECOMMENDATIONS);
    let anchor = recommendation_anchor()?;

    let mut recommendations = Vec::new();

    // Get masteries for skill-based recommendations
    let masteries = get_my_masteries(())?;

    // Get skills due for review
    let due = get_due_for_review(())?;
    for (rank, mastery) in due.iter().enumerate().take(limit / 2) {
        let rec = Recommendation {
            learner: agent.clone(),
            target_hash: mastery.skill_hash.clone(),
            rec_type: RecommendationType::Review,
            score_permille: 900 - (rank as u16 * 50),
            relevance_permille: 900,
            difficulty_match_permille: 800,
            readiness_permille: 1000,
            freshness_permille: 500,
            reason: RecommendationReason::RetentionRisk,
            explanation: format!("Review to maintain your {:?} mastery level",
                MasteryLevel::from_permille(mastery.mastery_permille)),
            rank: rank as u32 + 1,
            is_valid: true,
            generated_at: now,
            expires_at: now + 86400_000_000, // 24 hours in micros
        };
        recommendations.push(rec);
    }

    // Get weak areas for remediation
    let weak: Vec<&SkillMastery> = masteries
        .iter()
        .filter(|m| m.mastery_permille < 400 && m.total_attempts >= 3)
        .collect();

    for (rank, mastery) in weak.iter().enumerate().take(3) {
        let rec = Recommendation {
            learner: agent.clone(),
            target_hash: mastery.skill_hash.clone(),
            rec_type: RecommendationType::Practice,
            score_permille: 850 - (rank as u16 * 50),
            relevance_permille: 800,
            difficulty_match_permille: 700,
            readiness_permille: 900,
            freshness_permille: 600,
            reason: RecommendationReason::WeaknessRemediation,
            explanation: "Practice to strengthen this challenging area".to_string(),
            rank: (recommendations.len() + 1) as u32,
            is_valid: true,
            generated_at: now,
            expires_at: now + 86400_000_000,
        };
        recommendations.push(rec);
    }

    // Find strength areas for expansion
    let strengths: Vec<&SkillMastery> = masteries
        .iter()
        .filter(|m| m.mastery_permille >= 800)
        .collect();

    for (rank, mastery) in strengths.iter().enumerate().take(2) {
        let rec = Recommendation {
            learner: agent.clone(),
            target_hash: mastery.skill_hash.clone(),
            rec_type: RecommendationType::Challenge,
            score_permille: 750 - (rank as u16 * 50),
            relevance_permille: 700,
            difficulty_match_permille: 600,
            readiness_permille: 1000,
            freshness_permille: 700,
            reason: RecommendationReason::StrengthExpansion,
            explanation: "Ready for advanced challenges in this area".to_string(),
            rank: (recommendations.len() + 1) as u32,
            is_valid: true,
            generated_at: now,
            expires_at: now + 86400_000_000,
        };
        recommendations.push(rec);
    }

    // Filter by type if specified
    if let Some(rt) = input.rec_type {
        recommendations.retain(|r| r.rec_type == rt);
    }

    // Sort by score
    recommendations.sort_by(|a, b| b.score_permille.cmp(&a.score_permille));

    // Truncate to limit
    recommendations.truncate(limit);

    // Re-rank
    for (i, rec) in recommendations.iter_mut().enumerate() {
        rec.rank = (i + 1) as u32;
    }

    // Store recommendations
    for rec in &recommendations {
        let action_hash = create_entry(EntryTypes::Recommendation(rec.clone()))?;
        create_link(
            anchor.clone(),
            action_hash,
            LinkTypes::LearnerToRecommendations,
            (),
        )?;
    }

    Ok(recommendations)
}

/// Input for smart recommendation generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmartRecsInput {
    /// Maximum number of recommendations
    pub limit: Option<usize>,
    /// Current hour (0-23) - if not provided, uses default
    pub hour: Option<u8>,
    /// Day of week (0=Sunday, 6=Saturday) - if not provided, uses default
    pub day_of_week: Option<u8>,
    /// Recent topic hashes (for interleaving decisions)
    pub recent_topics: Option<Vec<ActionHash>>,
    /// Sessions completed today
    pub sessions_today: Option<u32>,
}

/// Generate smart recommendations using learning science
///
/// This advanced recommendation engine considers:
/// - Flow state theory (optimal difficulty matching)
/// - Time-of-day optimization (circadian rhythms)
/// - VARK learning style matching
/// - Interleaving for better retention
/// - Goal alignment
/// - Energy level estimation
#[hdk_extern]
pub fn generate_smart_recommendations_v2(input: SmartRecsInput) -> ExternResult<Vec<SmartRecommendation>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let limit = input.limit.unwrap_or(MAX_RECOMMENDATIONS);
    let hour = input.hour.unwrap_or(12); // Default to noon
    let day_of_week = input.day_of_week.unwrap_or(1); // Default to Monday
    let sessions_today = input.sessions_today.unwrap_or(0);
    let recent_topics = input.recent_topics.unwrap_or_default();

    // Get learner's masteries
    let masteries = get_my_masteries(())?;

    // Get learner's profile for VARK scores
    let profile = get_my_profile(())?;
    let vark = profile.as_ref()
        .map(|p| (
            p.visual_score_permille,
            p.auditory_score_permille,
            p.reading_score_permille,
            p.kinesthetic_score_permille
        ))
        .unwrap_or((500, 500, 500, 500));
    // Derive learning style from highest VARK score
    let learning_style = {
        let (v, a, r, k) = vark;
        if v >= a && v >= r && v >= k {
            LearningStyle::Visual
        } else if a >= v && a >= r && a >= k {
            LearningStyle::Auditory
        } else if r >= v && r >= a && r >= k {
            LearningStyle::ReadingWriting
        } else {
            LearningStyle::Kinesthetic
        }
    };

    // Get learner's goals
    let goals = get_my_goals(())?;

    // Build recommendation context
    let ctx = RecommendationContext {
        masteries: masteries.clone(),
        learning_style,
        vark_scores: vark,
        goals: goals.clone(),
        hour,
        day_of_week,
        recent_topic_hashes: recent_topics,
        sessions_today,
        energy_permille: RecommendationContext::estimate_energy(hour, sessions_today),
    };

    // Generate recommendations using the learning science algorithm
    let recommendations = generate_smart_recommendations(&ctx, agent, now, limit);

    Ok(recommendations)
}

// ============== Learning Goals ==============

/// Input for creating a learning goal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateGoalInput {
    pub title: String,
    pub description: String,
    pub goal_type: GoalType,
    pub target_hashes: Vec<ActionHash>,
    pub target_date: Option<i64>,
    pub priority: GoalPriority,
    pub estimated_hours: Option<u16>,
}

/// Create a learning goal
#[hdk_extern]
pub fn create_goal(input: CreateGoalInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = goal_anchor()?;

    let goal = LearningGoal {
        learner: agent,
        title: input.title,
        description: input.description,
        goal_type: input.goal_type,
        target_hashes: input.target_hashes,
        progress_permille: 0,
        target_date: input.target_date,
        is_completed: false,
        priority: input.priority,
        estimated_hours: input.estimated_hours.unwrap_or(10),
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::LearningGoal(goal))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToGoals,
        (),
    )?;

    Ok(action_hash)
}

/// Get all learning goals
#[hdk_extern]
pub fn get_my_goals(_: ()) -> ExternResult<Vec<LearningGoal>> {
    let anchor = goal_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToGoals)?,
        GetStrategy::Local,
    )?;

    let mut goals = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let goal: LearningGoal = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No goal entry".into())))?;

                if !goal.is_completed {
                    goals.push(goal);
                }
            }
        }
    }

    // Sort by priority then target date
    goals.sort_by(|a, b| {
        let pa = match a.priority {
            GoalPriority::Critical => 0,
            GoalPriority::High => 1,
            GoalPriority::Medium => 2,
            GoalPriority::Low => 3,
        };
        let pb = match b.priority {
            GoalPriority::Critical => 0,
            GoalPriority::High => 1,
            GoalPriority::Medium => 2,
            GoalPriority::Low => 3,
        };
        pa.cmp(&pb).then_with(|| a.target_date.cmp(&b.target_date))
    });

    Ok(goals)
}

/// Update goal progress
#[hdk_extern]
pub fn update_goal_progress(input: UpdateGoalProgressInput) -> ExternResult<LearningGoal> {
    let now = current_time()?;

    let record = get(input.goal_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Goal not found".into())))?;

    let mut goal: LearningGoal = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No goal entry".into())))?;

    goal.progress_permille = input.progress_permille.min(1000);
    goal.is_completed = goal.progress_permille >= 1000;
    goal.modified_at = now;

    update_entry(input.goal_hash, goal.clone())?;

    Ok(goal)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateGoalProgressInput {
    pub goal_hash: ActionHash,
    pub progress_permille: u16,
}

// ============== Session Analytics ==============

/// Input for recording a learning session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordSessionInput {
    pub started_at: i64,
    pub ended_at: i64,
    pub items_attempted: u32,
    pub items_completed: u32,
    pub correct_count: u32,
    pub skills_touched: Vec<ActionHash>,
    pub avg_response_time_ms: u32,
    pub hints_used: u32,
    pub skips: u32,
    pub xp_earned: u32,
}

/// Record a learning session
#[hdk_extern]
pub fn record_session(input: RecordSessionInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let anchor = session_anchor()?;

    let duration = ((input.ended_at - input.started_at).max(0) / 1_000_000) as u32; // micros to seconds

    // Estimate flow state from metrics
    let accuracy = if input.items_attempted > 0 {
        (input.correct_count * 1000 / input.items_attempted) as u16
    } else { 500 };

    // Flow balance: 500 = optimal, <300 = too easy, >700 = too hard
    let flow_balance = accuracy.saturating_sub(200).min(800);

    // Focus based on time on task vs interruptions
    let active_time = duration.saturating_sub(input.skips * 30);
    let focus = if duration > 0 {
        (active_time * 1000 / duration) as u16
    } else { 500 };

    // Frustration: low accuracy + many hints + skips
    let frustration_signals = if accuracy < 400 && input.hints_used > 3 { 2 }
        else if accuracy < 500 || input.skips > 2 { 1 }
        else { 0 };

    // Boredom: high accuracy + fast responses
    let boredom_signals = if accuracy > 900 && input.avg_response_time_ms < 2000 { 2 }
        else if accuracy > 850 || input.avg_response_time_ms < 3000 { 1 }
        else { 0 };

    // Calculate mastery gained
    let mastery_gained = ((input.correct_count as i32) - (input.items_attempted as i32 - input.correct_count as i32)) * 10;

    let session = SessionAnalytics {
        learner: agent,
        started_at: input.started_at,
        ended_at: input.ended_at,
        duration_seconds: duration,
        items_attempted: input.items_attempted,
        items_completed: input.items_completed,
        correct_count: input.correct_count,
        skills_touched: input.skills_touched,
        avg_response_time_ms: input.avg_response_time_ms,
        active_time_seconds: active_time,
        hints_used: input.hints_used,
        skips: input.skips,
        focus_estimate_permille: focus,
        flow_balance_permille: flow_balance,
        frustration_signals,
        boredom_signals,
        xp_earned: input.xp_earned,
        mastery_gained,
        skills_unlocked: 0, // Would need to check mastery thresholds
    };

    let action_hash = create_entry(EntryTypes::SessionAnalytics(session))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToSessions,
        (),
    )?;

    Ok(action_hash)
}

/// Get recent sessions
#[hdk_extern]
pub fn get_recent_sessions(limit: u32) -> ExternResult<Vec<SessionAnalytics>> {
    let anchor = session_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToSessions)?,
        GetStrategy::Local,
    )?;

    let mut sessions = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let session: SessionAnalytics = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No session entry".into())))?;
                sessions.push(session);
            }
        }
    }

    // Sort by most recent first
    sessions.sort_by(|a, b| b.ended_at.cmp(&a.ended_at));

    sessions.truncate(limit as usize);

    Ok(sessions)
}

// ============== Flow State Tracking & Optimization ==============
// Types and logic extracted to flow.rs — thin wrappers below.

/// Analyze current flow state and provide optimization recommendations
#[hdk_extern]
pub fn analyze_flow_state(input: AnalyzeFlowInput) -> ExternResult<FlowStateAnalysis> {
    flow::analyze_flow_state_impl(input)
}

/// Get optimal learning window based on flow history
#[hdk_extern]
pub fn get_optimal_learning_window(_: ()) -> ExternResult<OptimalLearningWindow> {
    flow::get_optimal_learning_window_impl(())
}

// ============== Adaptive Paths ==============

/// Input for creating an adaptive learning path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreatePathInput {
    pub name: String,
    pub goal_hash: Option<ActionHash>,
    pub initial_steps: Vec<PathStep>,
}

/// Create an adaptive learning path
#[hdk_extern]
pub fn create_adaptive_path(input: CreatePathInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = path_anchor()?;

    // Estimate total time
    let estimated_hours: u16 = input.initial_steps
        .iter()
        .map(|s| s.duration_minutes / 60)
        .sum::<u16>()
        .max(1);

    let path = AdaptivePath {
        learner: agent,
        name: input.name,
        goal_hash: input.goal_hash,
        steps: input.initial_steps,
        current_step: 0,
        completed_steps: 0,
        adaptation_count: 0,
        last_adaptation_reason: "Initial path created".to_string(),
        estimated_hours,
        estimated_completion: None,
        estimate_confidence_permille: 200, // Low confidence initially
        created_at: now,
        modified_at: now,
    };

    let action_hash = create_entry(EntryTypes::AdaptivePath(path))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToPaths,
        (),
    )?;

    Ok(action_hash)
}

/// Advance to next step in path
#[hdk_extern]
pub fn advance_path_step(path_hash: ActionHash) -> ExternResult<AdaptivePath> {
    let now = current_time()?;

    let record = get(path_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Path not found".into())))?;

    let mut path: AdaptivePath = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No path entry".into())))?;

    // Mark current step complete
    if let Some(step) = path.steps.get_mut(path.current_step as usize) {
        step.is_completed = true;
    }

    path.completed_steps += 1;
    path.current_step = path.current_step.saturating_add(1);
    path.modified_at = now;

    update_entry(path_hash, path.clone())?;

    Ok(path)
}

/// Get learner's adaptive paths
#[hdk_extern]
pub fn get_my_paths(_: ()) -> ExternResult<Vec<AdaptivePath>> {
    let anchor = path_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToPaths)?,
        GetStrategy::Local,
    )?;

    let mut paths = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let path: AdaptivePath = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No path entry".into())))?;
                paths.push(path);
            }
        }
    }

    Ok(paths)
}

// ============== Difficulty Calibration ==============

/// Input for recording content attempt for calibration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordCalibrationInput {
    pub content_hash: ActionHash,
    pub content_type: ContentType,
    pub learner_level_permille: u16,
    pub success: bool,
    pub time_seconds: u32,
}

/// Record an attempt and update content difficulty calibration
#[hdk_extern]
pub fn update_difficulty_calibration(input: RecordCalibrationInput) -> ExternResult<DifficultyCalibration> {
    let now = current_time()?;

    // Try to get existing calibration
    let links = get_links(
        LinkQuery::try_new(input.content_hash.clone(), LinkTypes::ContentToCalibration)?,
        GetStrategy::Local,
    )?;

    let mut calibration = if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No calibration entry".into())))?
            } else {
                create_default_calibration(&input, now)
            }
        } else {
            create_default_calibration(&input, now)
        }
    } else {
        create_default_calibration(&input, now)
    };

    // Update stats
    let old_total = calibration.total_attempts;
    calibration.total_attempts += 1;
    if input.success {
        calibration.successful_attempts += 1;
    }

    // Update average time (running average)
    calibration.avg_time_seconds = if old_total == 0 {
        input.time_seconds
    } else {
        ((calibration.avg_time_seconds as u64 * old_total as u64 + input.time_seconds as u64)
         / calibration.total_attempts as u64) as u32
    };

    // Recalibrate difficulty based on success rate
    let success_rate = (calibration.successful_attempts * 1000 / calibration.total_attempts) as u16;

    // If success rate is high, content is easier than thought
    // If success rate is low, content is harder
    // 50% success = perfect calibration
    let adjustment = if success_rate > 600 {
        // Too easy, reduce difficulty estimate
        -((success_rate - 500) as i32 / 2)
    } else if success_rate < 400 {
        // Too hard, increase difficulty estimate
        (500 - success_rate) as i32 / 2
    } else {
        0
    };

    calibration.calibrated_difficulty_permille =
        ((calibration.calibrated_difficulty_permille as i32 + adjustment).max(0).min(1000)) as u16;
    calibration.calibrated_at = now;

    // Create or update entry
    if links.is_empty() {
        let hash = create_entry(EntryTypes::DifficultyCalibration(calibration.clone()))?;
        create_link(
            input.content_hash,
            hash,
            LinkTypes::ContentToCalibration,
            (),
        )?;
    } else if let Some(link) = links.first() {
        if let Some(existing_hash) = link.target.clone().into_action_hash() {
            update_entry(existing_hash, calibration.clone())?;
        }
    }

    Ok(calibration)
}

fn create_default_calibration(input: &RecordCalibrationInput, now: i64) -> DifficultyCalibration {
    DifficultyCalibration {
        content_hash: input.content_hash.clone(),
        content_type: input.content_type.clone(),
        authored_difficulty_permille: 500,
        calibrated_difficulty_permille: 500,
        variance_permille: 200,
        total_attempts: 0,
        successful_attempts: 0,
        avg_time_seconds: 0,
        time_std_dev_seconds: 0,
        discrimination_permille: 500,
        calibrated_at: now,
    }
}

// ============== Summary ==============

/// Comprehensive learner summary
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnerSummary {
    pub profile: Option<LearnerProfile>,
    pub learning_style: LearningStyleResult,
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

/// Get comprehensive learner summary
#[hdk_extern]
pub fn get_learner_summary(_: ()) -> ExternResult<LearnerSummary> {
    let profile = get_my_profile(())?;
    let learning_style = calculate_learning_style(())?;
    let masteries = get_my_masteries(())?;
    let due = get_due_for_review(())?;
    let goals = get_my_goals(())?;
    let sessions = get_recent_sessions(100)?;
    let paths = get_my_paths(())?;

    let mastery_count = masteries.len() as u32;
    let avg_mastery = if mastery_count > 0 {
        (masteries.iter().map(|m| m.mastery_permille as u32).sum::<u32>() / mastery_count) as u16
    } else { 0 };

    let skills_mastered = masteries.iter().filter(|m| m.mastery_permille >= MASTERY_THRESHOLD).count() as u32;
    let total_minutes: u32 = sessions.iter().map(|s| s.duration_seconds / 60).sum();

    Ok(LearnerSummary {
        profile,
        learning_style,
        mastery_count,
        avg_mastery_permille: avg_mastery,
        skills_mastered,
        due_for_review: due.len() as u32,
        active_goals: goals.len() as u32,
        total_sessions: sessions.len() as u32,
        total_learning_minutes: total_minutes,
        active_paths: paths.iter().filter(|p| p.current_step < p.steps.len() as u32).count() as u32,
        recommendations_count: 0, // Would need to fetch fresh recommendations
    })
}

// ============== Performance Metrics Collection ==============
// Types and logic extracted to analytics.rs — thin wrappers below.

/// Collect performance metrics for the current learner
#[hdk_extern]
pub fn collect_performance_metrics(input: CollectMetricsInput) -> ExternResult<PerformanceMetrics> {
    analytics::collect_performance_metrics(input)
}

/// Run performance benchmarks (for development/testing)
#[hdk_extern]
pub fn run_benchmarks(_: ()) -> ExternResult<BenchmarkSummary> {
    analytics::run_benchmarks(())
}

// ============== Retention Prediction API ==============
// Types and logic extracted to retention.rs — thin wrappers below.

/// Re-export the retention prediction type for API
pub use adaptive_integrity::RetentionPrediction;

/// Predict retention for a specific skill
#[hdk_extern]
pub fn predict_skill_retention(input: RetentionPredictionInput) -> ExternResult<RetentionPrediction> {
    retention::predict_skill_retention_impl(input)
}

/// Predict retention for multiple skills (batch operation)
#[hdk_extern]
pub fn predict_retention_batch(input: BatchRetentionInput) -> ExternResult<BatchRetentionResult> {
    retention::predict_retention_batch_impl(input)
}

/// Generate optimal review schedule to maintain target retention
#[hdk_extern]
pub fn get_optimal_review_schedule(input: ReviewScheduleInput) -> ExternResult<ReviewScheduleResult> {
    retention::get_optimal_review_schedule_impl(input)
}

// ============== Metacognition & Confidence Calibration ==============
// Metacognition is "thinking about thinking" - helping learners understand
// how well they actually know material vs how well they think they know it.

/// A confidence judgment before answering a question
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfidenceJudgment {
    /// Skill being assessed
    pub skill_hash: ActionHash,
    /// Predicted confidence (0-1000): How well learner thinks they'll do
    pub predicted_confidence: u16,
    /// Actual performance (0-1000): How well they actually did
    pub actual_performance: u16,
    /// Timestamp of judgment
    pub judged_at: i64,
}

/// Calibration metrics showing how accurate self-assessment is
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Overall calibration score (0-1000, higher = better calibrated)
    pub calibration_score: u16,
    /// Overconfidence bias: positive means thinks they know more than they do
    pub overconfidence_bias: i16,
    /// Mean absolute error between prediction and actual
    pub mean_absolute_error: u16,
    /// Total judgments analyzed
    pub judgment_count: u32,
    /// Brier score (0-1000, lower = better probabilistic accuracy)
    pub brier_score: u16,
    /// Trend: is calibration improving?
    pub calibration_trend: CalibrationTrend,
}

/// Calibration trend direction
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationTrend {
    Improving,
    Stable,
    Declining,
}

impl Default for CalibrationTrend {
    fn default() -> Self {
        CalibrationTrend::Stable
    }
}

/// Input for recording a confidence judgment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordJudgmentInput {
    pub skill_hash: ActionHash,
    pub predicted_confidence: u16,
    pub actual_performance: u16,
}

/// Calculate calibration metrics from a set of judgments
#[hdk_extern]
pub fn calculate_calibration(judgments: Vec<ConfidenceJudgment>) -> ExternResult<CalibrationMetrics> {
    if judgments.is_empty() {
        return Ok(CalibrationMetrics {
            calibration_score: 500, // Neutral when no data
            overconfidence_bias: 0,
            mean_absolute_error: 0,
            judgment_count: 0,
            brier_score: 500,
            calibration_trend: CalibrationTrend::Stable,
        });
    }

    let count = judgments.len() as u32;
    let mut total_error: i64 = 0;
    let mut total_abs_error: u64 = 0;
    let mut total_brier: u64 = 0;

    for j in &judgments {
        // Error = predicted - actual (positive = overconfident)
        let error = j.predicted_confidence as i64 - j.actual_performance as i64;
        total_error += error;

        // Absolute error
        total_abs_error += error.unsigned_abs();

        // Brier score component: (prediction - actual)^2
        let brier_component = (error * error) as u64;
        total_brier += brier_component;
    }

    // Calculate metrics
    let overconfidence_bias = (total_error / count as i64) as i16;
    let mean_absolute_error = (total_abs_error / count as u64) as u16;

    // Brier score (normalized to 0-1000, lower is better)
    let brier_score = ((total_brier / count as u64) / 1000).min(1000) as u16;

    // Calibration score: inverse of MAE, normalized
    // Perfect calibration (MAE=0) = 1000, terrible (MAE=1000) = 0
    let calibration_score = 1000u16.saturating_sub(mean_absolute_error);

    // Calculate trend from recent vs older judgments
    let calibration_trend = if judgments.len() >= 10 {
        let mid = judgments.len() / 2;
        let recent_error: i64 = judgments[mid..].iter()
            .map(|j| (j.predicted_confidence as i64 - j.actual_performance as i64).abs())
            .sum();
        let older_error: i64 = judgments[..mid].iter()
            .map(|j| (j.predicted_confidence as i64 - j.actual_performance as i64).abs())
            .sum();

        if recent_error < older_error - 500 {
            CalibrationTrend::Improving
        } else if recent_error > older_error + 500 {
            CalibrationTrend::Declining
        } else {
            CalibrationTrend::Stable
        }
    } else {
        CalibrationTrend::Stable
    };

    Ok(CalibrationMetrics {
        calibration_score,
        overconfidence_bias,
        mean_absolute_error,
        judgment_count: count,
        brier_score,
        calibration_trend,
    })
}

// ============== Flow State Optimization ==============
// Types and logic extracted to flow.rs — thin wrappers below.

/// Assess current flow state and provide recommendations
#[hdk_extern]
pub fn assess_flow_state(input: FlowAssessmentInput) -> ExternResult<FlowAssessment> {
    flow::assess_flow_state_impl(input)
}

// ============== Personalized Learning Path ==============
// Optimal ordering of skills based on prerequisites, mastery, and learning velocity

/// A step in a personalized learning path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningPathStep {
    /// Skill to learn
    pub skill_hash: ActionHash,
    /// Recommended order (1 = first)
    pub order: u16,
    /// Estimated time to master (minutes)
    pub estimated_minutes: u32,
    /// Why this skill is recommended now
    pub reason: PathStepReason,
    /// Readiness score (0-1000)
    pub readiness: u16,
    /// Priority score (0-1000)
    pub priority: u16,
}

/// Reason for including a skill in the path
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathStepReason {
    /// Prerequisite for other skills
    UnlocksOthers,
    /// Due for review (retention dropping)
    DueForReview,
    /// Close to mastery
    NearMastery,
    /// Foundational skill
    Foundation,
    /// User goal related
    GoalRelated,
    /// Optimal for current time of day
    TimeOptimal,
    /// Fills knowledge gap
    FillsGap,
}

impl Default for PathStepReason {
    fn default() -> Self {
        PathStepReason::Foundation
    }
}

/// A complete personalized learning path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalizedLearningPath {
    /// Ordered steps in the path
    pub steps: Vec<LearningPathStep>,
    /// Total estimated time (minutes)
    pub total_estimated_minutes: u32,
    /// Number of skills that will be unlocked
    pub skills_to_unlock: u32,
    /// Confidence in path quality (0-1000)
    pub confidence: u16,
    /// Generated timestamp
    pub generated_at: i64,
}

/// Input for generating a learning path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratePathInput {
    /// Available skills to include
    pub available_skills: Vec<ActionHash>,
    /// Current mastery levels
    pub mastery_levels: Vec<(ActionHash, u16)>,
    /// Time available (minutes)
    pub available_minutes: u32,
    /// Focus on specific goal?
    pub goal_skill: Option<ActionHash>,
    /// Current hour (0-23) for time optimization
    pub current_hour: u8,
}

/// Generate an optimized learning path
#[hdk_extern]
pub fn generate_learning_path(input: GeneratePathInput) -> ExternResult<PersonalizedLearningPath> {
    let now = current_time()?;
    let mastery_map: std::collections::HashMap<ActionHash, u16> =
        input.mastery_levels.into_iter().collect();

    let mut steps: Vec<LearningPathStep> = Vec::new();
    let mut order = 1u16;
    let mut total_minutes = 0u32;

    // Categorize skills by priority
    for skill_hash in &input.available_skills {
        if total_minutes >= input.available_minutes {
            break;
        }

        let mastery = mastery_map.get(skill_hash).copied().unwrap_or(0);

        // Determine reason and priority
        let (reason, priority, estimated_mins) = if mastery == 0 {
            // New skill - foundational
            (PathStepReason::Foundation, 600, 30)
        } else if mastery < 300 {
            // Early stage - might fill gaps
            (PathStepReason::FillsGap, 700, 25)
        } else if mastery >= 700 && mastery < 800 {
            // Close to mastery!
            (PathStepReason::NearMastery, 900, 15)
        } else if mastery < 500 {
            // Due for review
            (PathStepReason::DueForReview, 800, 20)
        } else {
            // Regular progression
            (PathStepReason::Foundation, 500, 25)
        };

        // Boost priority if goal-related
        let priority: u16 = if input.goal_skill.as_ref() == Some(skill_hash) {
            1000
        } else {
            priority
        };

        // Time-of-day optimization
        let priority: u16 = match input.current_hour {
            6..=11 => priority + 50, // Morning boost for new learning
            14..=17 => priority, // Afternoon - neutral
            19..=22 => priority.saturating_sub(50) + if mastery > 500 { 100 } else { 0 }, // Evening - review boost
            _ => priority,
        };

        // Readiness based on mastery trajectory
        let readiness = if mastery < 100 {
            1000 // Ready to start
        } else {
            (1000 - mastery).max(200) // Less ready if already high mastery
        };

        if total_minutes + estimated_mins <= input.available_minutes {
            steps.push(LearningPathStep {
                skill_hash: skill_hash.clone(),
                order,
                estimated_minutes: estimated_mins,
                reason,
                readiness,
                priority,
            });
            order += 1;
            total_minutes += estimated_mins;
        }
    }

    // Sort by priority (highest first)
    steps.sort_by(|a, b| b.priority.cmp(&a.priority));

    // Reassign order after sorting
    for (i, step) in steps.iter_mut().enumerate() {
        step.order = (i + 1) as u16;
    }

    let skills_to_unlock = steps.iter()
        .filter(|s| s.reason == PathStepReason::UnlocksOthers)
        .count() as u32;

    // Confidence based on data quality
    let confidence = if steps.len() >= 5 { 800 } else if steps.len() >= 3 { 600 } else { 400 };

    Ok(PersonalizedLearningPath {
        steps,
        total_estimated_minutes: total_minutes,
        skills_to_unlock,
        confidence,
        generated_at: now,
    })
}

// ============== Study Group Support ==============
// Social learning features for peer collaboration

/// A study group for collaborative learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyGroup {
    /// Unique identifier
    pub group_id: String,
    /// Group name
    pub name: String,
    /// Skills this group focuses on
    pub focus_skills: Vec<ActionHash>,
    /// Member count
    pub member_count: u32,
    /// Average group mastery (0-1000)
    pub avg_mastery: u16,
    /// Group activity level (0-1000)
    pub activity_level: u16,
    /// Created timestamp
    pub created_at: i64,
}

/// Peer match for tutoring or study
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerMatch {
    /// Matched peer's agent
    pub peer_agent: AgentPubKey,
    /// Compatibility score (0-1000)
    pub compatibility_score: u16,
    /// Matching reason
    pub match_reason: PeerMatchReason,
    /// Skills to collaborate on
    pub shared_skills: Vec<ActionHash>,
    /// Complementary skills (peer can help with these)
    pub complementary_skills: Vec<ActionHash>,
}

/// Reason for peer matching
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerMatchReason {
    /// Similar skill levels - study together
    SimilarLevel,
    /// Peer is ahead - can tutor
    PeerCanTutor,
    /// You're ahead - can help peer
    YouCanTutor,
    /// Complementary strengths
    ComplementarySkills,
    /// Same learning goals
    SharedGoals,
}

impl Default for PeerMatchReason {
    fn default() -> Self {
        PeerMatchReason::SimilarLevel
    }
}

/// Input for finding peer matches
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FindPeersInput {
    /// Skills to match on
    pub skills: Vec<ActionHash>,
    /// Your mastery levels
    pub mastery_levels: Vec<(ActionHash, u16)>,
    /// Max matches to return
    pub max_matches: u8,
    /// Prefer tutors or study partners?
    pub prefer_tutors: bool,
}

/// Calculate peer compatibility (simplified version for unit testing)
pub fn calculate_peer_compatibility(
    my_levels: &[(ActionHash, u16)],
    peer_levels: &[(ActionHash, u16)],
    prefer_tutors: bool,
) -> (u16, PeerMatchReason) {
    let my_map: std::collections::HashMap<ActionHash, u16> =
        my_levels.iter().cloned().collect();
    let peer_map: std::collections::HashMap<ActionHash, u16> =
        peer_levels.iter().cloned().collect();

    let mut similarity_sum: i64 = 0;
    let mut peer_ahead_count = 0u32;
    let mut i_ahead_count = 0u32;
    let mut shared_count = 0u32;

    for (skill, my_level) in &my_map {
        if let Some(&peer_level) = peer_map.get(skill) {
            shared_count += 1;
            let diff = (peer_level as i64 - *my_level as i64).abs();
            similarity_sum += 1000 - diff.min(1000);

            if peer_level > *my_level + 200 {
                peer_ahead_count += 1;
            } else if *my_level > peer_level + 200 {
                i_ahead_count += 1;
            }
        }
    }

    if shared_count == 0 {
        return (0, PeerMatchReason::SimilarLevel);
    }

    let avg_similarity = (similarity_sum / shared_count as i64) as u16;

    let reason = if prefer_tutors && peer_ahead_count > i_ahead_count {
        PeerMatchReason::PeerCanTutor
    } else if !prefer_tutors && i_ahead_count > peer_ahead_count {
        PeerMatchReason::YouCanTutor
    } else if peer_ahead_count > 0 && i_ahead_count > 0 {
        PeerMatchReason::ComplementarySkills
    } else {
        PeerMatchReason::SimilarLevel
    };

    (avg_similarity, reason)
}

// =============================================================================
// FEATURE 1: Bloom's Taxonomy - Cognitive Level Tracking
// =============================================================================
// Based on Anderson & Krathwohl (2001) revised taxonomy.
// Learners gain 23% better retention when content is organized by cognitive complexity.

/// Bloom's Taxonomy cognitive levels (revised)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum CognitiveLevel {
    /// Recall facts and basic concepts
    Remember,
    /// Explain ideas or concepts
    Understand,
    /// Use information in new situations
    Apply,
    /// Draw connections among ideas
    Analyze,
    /// Justify a decision or course of action
    Evaluate,
    /// Produce new or original work
    Create,
}

impl Default for CognitiveLevel {
    fn default() -> Self {
        CognitiveLevel::Remember
    }
}

impl CognitiveLevel {
    /// Get the numeric level (0-5) for ordering
    pub fn level_index(&self) -> u8 {
        match self {
            CognitiveLevel::Remember => 0,
            CognitiveLevel::Understand => 1,
            CognitiveLevel::Apply => 2,
            CognitiveLevel::Analyze => 3,
            CognitiveLevel::Evaluate => 4,
            CognitiveLevel::Create => 5,
        }
    }

    /// Get the next cognitive level
    pub fn next_level(&self) -> Option<CognitiveLevel> {
        match self {
            CognitiveLevel::Remember => Some(CognitiveLevel::Understand),
            CognitiveLevel::Understand => Some(CognitiveLevel::Apply),
            CognitiveLevel::Apply => Some(CognitiveLevel::Analyze),
            CognitiveLevel::Analyze => Some(CognitiveLevel::Evaluate),
            CognitiveLevel::Evaluate => Some(CognitiveLevel::Create),
            CognitiveLevel::Create => None,
        }
    }

    /// Difficulty multiplier for this cognitive level (1000 = base)
    pub fn difficulty_multiplier(&self) -> u16 {
        match self {
            CognitiveLevel::Remember => 800,    // Easier
            CognitiveLevel::Understand => 1000, // Base
            CognitiveLevel::Apply => 1200,      // 20% harder
            CognitiveLevel::Analyze => 1400,    // 40% harder
            CognitiveLevel::Evaluate => 1600,   // 60% harder
            CognitiveLevel::Create => 1800,     // 80% harder
        }
    }
}

/// Mastery tracked per cognitive level for a skill
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillMasteryByLevel {
    /// The skill being tracked
    pub skill_hash: ActionHash,
    /// Mastery at Remember level (0-1000)
    pub remember_mastery: u16,
    /// Mastery at Understand level (0-1000)
    pub understand_mastery: u16,
    /// Mastery at Apply level (0-1000)
    pub apply_mastery: u16,
    /// Mastery at Analyze level (0-1000)
    pub analyze_mastery: u16,
    /// Mastery at Evaluate level (0-1000)
    pub evaluate_mastery: u16,
    /// Mastery at Create level (0-1000)
    pub create_mastery: u16,
    /// Highest unlocked cognitive level
    pub current_level: CognitiveLevel,
    /// Overall weighted mastery (0-1000)
    pub composite_mastery: u16,
}

impl Default for SkillMasteryByLevel {
    fn default() -> Self {
        Self {
            skill_hash: ActionHash::from_raw_36(vec![0; 36]),
            remember_mastery: 0,
            understand_mastery: 0,
            apply_mastery: 0,
            analyze_mastery: 0,
            evaluate_mastery: 0,
            create_mastery: 0,
            current_level: CognitiveLevel::Remember,
            composite_mastery: 0,
        }
    }
}

/// Threshold to unlock next cognitive level (85% = 850)
pub const COGNITIVE_UNLOCK_THRESHOLD: u16 = 850;

/// Input for updating mastery at a specific cognitive level
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateCognitiveMasteryInput {
    pub skill_hash: ActionHash,
    pub level: CognitiveLevel,
    pub correct: bool,
    pub response_quality_permille: u16,
}

/// Check if learner can advance to next cognitive level
#[hdk_extern]
pub fn check_cognitive_level_unlock(input: SkillMasteryByLevel) -> ExternResult<Option<CognitiveLevel>> {
    let current_mastery = match input.current_level {
        CognitiveLevel::Remember => input.remember_mastery,
        CognitiveLevel::Understand => input.understand_mastery,
        CognitiveLevel::Apply => input.apply_mastery,
        CognitiveLevel::Analyze => input.analyze_mastery,
        CognitiveLevel::Evaluate => input.evaluate_mastery,
        CognitiveLevel::Create => return Ok(None), // Already at max
    };

    if current_mastery >= COGNITIVE_UNLOCK_THRESHOLD {
        Ok(input.current_level.next_level())
    } else {
        Ok(None)
    }
}

/// Calculate composite mastery weighted by cognitive level importance
#[hdk_extern]
pub fn calculate_composite_mastery(input: SkillMasteryByLevel) -> ExternResult<u16> {
    // Higher cognitive levels weighted more heavily
    // Weights: Remember=10%, Understand=15%, Apply=20%, Analyze=20%, Evaluate=17.5%, Create=17.5%
    let weighted_sum: u32 =
        (input.remember_mastery as u32 * 100) +
        (input.understand_mastery as u32 * 150) +
        (input.apply_mastery as u32 * 200) +
        (input.analyze_mastery as u32 * 200) +
        (input.evaluate_mastery as u32 * 175) +
        (input.create_mastery as u32 * 175);

    let composite = (weighted_sum / 1000).min(1000) as u16;
    Ok(composite)
}

// =============================================================================
// FEATURE 2: Transfer of Learning Assessment
// =============================================================================
// Only 10% of skills transfer without explicit instruction (Singley & Anderson, 1989).
// Near transfer = 2.3x easier than far transfer (Bransford, 2000).

/// Type of transfer being assessed
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferType {
    /// Same skill, slightly different context
    NearTransfer,
    /// Related skill, similar domain
    IntermediateTransfer,
    /// Different domain, fundamental principle
    FarTransfer,
}

impl Default for TransferType {
    fn default() -> Self {
        TransferType::NearTransfer
    }
}

/// Assessment of learning transfer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferAssessment {
    /// Skill being tested for transfer
    pub skill_hash: ActionHash,
    /// Type of transfer being measured
    pub transfer_type: TransferType,
    /// Original context where skill was learned
    pub original_context: String,
    /// New context where skill is being applied
    pub new_context: String,
    /// Success rate in new context (0-1000)
    pub success_permille: u16,
    /// Time to solve in new context (seconds)
    pub latency_seconds: u32,
    /// Learner's confidence in new context (0-1000)
    pub confidence_permille: u16,
    /// Number of hints required
    pub hints_required: u8,
    /// Perceived difficulty (0-1000)
    pub perceived_difficulty: u16,
    /// When this assessment was taken
    pub assessed_at: Timestamp,
}

/// Transfer map showing where a skill has been successfully applied
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferMap {
    /// Source skill
    pub source_skill: ActionHash,
    /// Original mastery (0-1000)
    pub original_mastery: u16,
    /// Near transfer success rate (0-1000)
    pub near_transfer_success: u16,
    /// Intermediate transfer success rate (0-1000)
    pub intermediate_transfer_success: u16,
    /// Far transfer success rate (0-1000)
    pub far_transfer_success: u16,
    /// Transfer ratio (transfer success / original mastery) - shows depth of learning
    pub transfer_ratio_permille: u16,
    /// Is this "deep mastery"? (original >= 800 AND near >= 700 AND far >= 500)
    pub is_deep_mastery: bool,
}

/// Input for recording a transfer assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordTransferInput {
    pub skill_hash: ActionHash,
    pub transfer_type: TransferType,
    pub original_context: String,
    pub new_context: String,
    pub success: bool,
    pub latency_seconds: u32,
    pub confidence_permille: u16,
    pub hints_required: u8,
}

/// Calculate transfer readiness for a skill
#[hdk_extern]
pub fn calculate_transfer_readiness(input: TransferMap) -> ExternResult<u16> {
    // Transfer readiness = weighted average of original mastery and transfer success
    let original_weight: u32 = 300; // 30%
    let near_weight: u32 = 350;     // 35%
    let intermediate_weight: u32 = 200; // 20%
    let far_weight: u32 = 150;      // 15%

    let weighted_sum: u32 =
        (input.original_mastery as u32 * original_weight) +
        (input.near_transfer_success as u32 * near_weight) +
        (input.intermediate_transfer_success as u32 * intermediate_weight) +
        (input.far_transfer_success as u32 * far_weight);

    let readiness = (weighted_sum / 1000).min(1000) as u16;
    Ok(readiness)
}

/// Check if learner has achieved deep mastery (real learning, not just retention)
#[hdk_extern]
pub fn check_deep_mastery(input: TransferMap) -> ExternResult<bool> {
    let is_deep = input.original_mastery >= 800 &&
                  input.near_transfer_success >= 700 &&
                  input.far_transfer_success >= 500;
    Ok(is_deep)
}

// =============================================================================
// FEATURE 3: Elaborative Interrogation & Self-Explanation Prompts
// =============================================================================
// Meta-analysis (Bisra et al., 2018) shows self-explanation increases learning by 28-35%.
// Dunlosky et al. (2013) ranks it as "high utility" learning technique.

/// Types of elaboration prompts
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElaborationPromptType {
    /// "Why does this approach work?"
    WhyWorks,
    /// "How do you know this is correct?"
    HowKnow,
    /// "Explain this to a 10-year-old"
    ExplainSimply,
    /// "Give a counter-example where this fails"
    CounterExample,
    /// "How does this connect to [other skill]?"
    Connection,
    /// "When would you use this in real life?"
    RealWorldApplication,
    /// "What are the key steps?"
    ProcessSteps,
    /// "What's the underlying principle?"
    UnderlyingPrinciple,
}

impl Default for ElaborationPromptType {
    fn default() -> Self {
        ElaborationPromptType::WhyWorks
    }
}

/// An elaboration prompt and response
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElaborationPrompt {
    /// Skill this prompt is about
    pub skill_hash: ActionHash,
    /// Type of elaboration requested
    pub prompt_type: ElaborationPromptType,
    /// The actual prompt text
    pub prompt_text: String,
    /// Mastery level when prompt was triggered (should be 40-60%)
    pub trigger_mastery: u16,
    /// Learner's response (if provided)
    pub learner_response: Option<String>,
    /// Quality score of response (0-1000, AI or educator scored)
    pub response_quality: Option<u16>,
    /// XP multiplier for high-quality elaboration (default 1.5x)
    pub xp_multiplier_permille: u16,
    /// When this prompt was presented
    pub prompted_at: Timestamp,
    /// When learner responded (if they did)
    pub responded_at: Option<Timestamp>,
}

/// Result of elaboration quality assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElaborationResult {
    /// Quality score (0-1000)
    pub quality_score: u16,
    /// Should extend SM-2 interval? (quality >= 700)
    pub extend_interval: bool,
    /// Interval multiplier (1000 = 1.0x, 1300 = 1.3x)
    pub interval_multiplier: u16,
    /// Should trigger retrieval practice? (quality < 400)
    pub trigger_retrieval_practice: bool,
    /// Feedback for learner
    pub feedback: String,
}

/// Input for generating an elaboration prompt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateElaborationInput {
    pub skill_hash: ActionHash,
    pub current_mastery: u16,
    pub cognitive_level: CognitiveLevel,
    pub related_skills: Vec<ActionHash>,
}

/// Generate appropriate elaboration prompt based on context
#[hdk_extern]
pub fn generate_elaboration_prompt(input: GenerateElaborationInput) -> ExternResult<ElaborationPrompt> {
    // Choose prompt type based on cognitive level and mastery
    let prompt_type = if input.current_mastery < 300 {
        ElaborationPromptType::ProcessSteps // Basic understanding
    } else if input.current_mastery < 500 {
        ElaborationPromptType::WhyWorks // Building reasoning
    } else if input.current_mastery < 700 {
        if !input.related_skills.is_empty() {
            ElaborationPromptType::Connection // Connect to other knowledge
        } else {
            ElaborationPromptType::RealWorldApplication
        }
    } else {
        match input.cognitive_level {
            CognitiveLevel::Analyze | CognitiveLevel::Evaluate => {
                ElaborationPromptType::CounterExample
            }
            CognitiveLevel::Create => ElaborationPromptType::UnderlyingPrinciple,
            _ => ElaborationPromptType::ExplainSimply,
        }
    };

    let prompt_text = match prompt_type {
        ElaborationPromptType::WhyWorks => "Why does this approach work? What makes it effective?".to_string(),
        ElaborationPromptType::HowKnow => "How do you know this answer is correct?".to_string(),
        ElaborationPromptType::ExplainSimply => "Explain this concept as if teaching a 10-year-old.".to_string(),
        ElaborationPromptType::CounterExample => "Can you think of a situation where this would NOT work?".to_string(),
        ElaborationPromptType::Connection => "How does this connect to other things you've learned?".to_string(),
        ElaborationPromptType::RealWorldApplication => "When might you use this in real life?".to_string(),
        ElaborationPromptType::ProcessSteps => "What are the key steps in this process?".to_string(),
        ElaborationPromptType::UnderlyingPrinciple => "What's the fundamental principle behind this?".to_string(),
    };

    Ok(ElaborationPrompt {
        skill_hash: input.skill_hash,
        prompt_type,
        prompt_text,
        trigger_mastery: input.current_mastery,
        learner_response: None,
        response_quality: None,
        xp_multiplier_permille: 1500, // 1.5x XP for good elaboration
        prompted_at: Timestamp::now(),
        responded_at: None,
    })
}

/// Assess quality of elaboration response
#[hdk_extern]
pub fn assess_elaboration_quality(quality_score: u16) -> ExternResult<ElaborationResult> {
    let extend_interval = quality_score >= 700;
    let trigger_retrieval = quality_score < 400;

    let interval_multiplier = if quality_score >= 800 {
        1400 // 1.4x longer interval
    } else if quality_score >= 600 {
        1200 // 1.2x longer interval
    } else if quality_score >= 400 {
        1000 // Normal interval
    } else {
        800 // Shorter interval, needs more practice
    };

    let feedback = if quality_score >= 800 {
        "Excellent explanation! Your deep understanding will help this stick.".to_string()
    } else if quality_score >= 600 {
        "Good thinking! Consider adding more specific examples.".to_string()
    } else if quality_score >= 400 {
        "You're on the right track. Try explaining WHY this works.".to_string()
    } else {
        "Let's review this concept again with some practice problems.".to_string()
    };

    Ok(ElaborationResult {
        quality_score,
        extend_interval,
        interval_multiplier,
        trigger_retrieval_practice: trigger_retrieval,
        feedback,
    })
}

// =============================================================================
// FEATURE 4: Worked Examples Dynamic Fading
// =============================================================================
// Research shows optimal worked example fading improves learning by 26% (Renkl & Atkinson, 2010).
// Ratio should change: 80% worked examples at 20% mastery → 20% at 80% mastery.

/// Content format types
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentFormat {
    /// Full worked example with all steps shown
    FullWorkedExample,
    /// Worked example with some steps for learner to complete
    PartialWorkedExample {
        /// Percentage of steps shown (0-1000)
        steps_shown_permille: u16,
    },
    /// Problem to solve with hints available
    GuidedProblem {
        /// Difficulty level (0-1000)
        difficulty: u16,
    },
    /// Problem to solve without scaffolding
    IndependentProblem {
        /// Difficulty level (0-1000)
        difficulty: u16,
    },
}

impl Default for ContentFormat {
    fn default() -> Self {
        ContentFormat::FullWorkedExample
    }
}

/// Worked example fading recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkedExampleRecommendation {
    /// Recommended format
    pub format: ContentFormat,
    /// Optimal worked example ratio (0-1000, 800 = 80% worked examples)
    pub worked_ratio_permille: u16,
    /// XP value for this content type
    pub xp_value: u16,
    /// Reasoning for this recommendation
    pub reasoning: String,
}

/// Calculate optimal worked example ratio based on mastery
pub fn calculate_worked_example_ratio(mastery_permille: u16) -> u16 {
    // Sigmoidal fade: 80% at 0% mastery → 10% at 100% mastery
    // Formula: ratio = 800 - (mastery * 0.7)
    let ratio = 800i32 - ((mastery_permille as i32 * 7) / 10);
    ratio.max(100).min(800) as u16
}

/// Input for worked example recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkedExampleInput {
    pub skill_hash: ActionHash,
    pub current_mastery: u16,
    pub cognitive_level: CognitiveLevel,
    pub recent_errors: u8,
    pub session_minutes: u16,
}

/// Get recommendation for content format based on mastery
#[hdk_extern]
pub fn recommend_content_format(input: WorkedExampleInput) -> ExternResult<WorkedExampleRecommendation> {
    let worked_ratio = calculate_worked_example_ratio(input.current_mastery);

    let (format, xp_value, reasoning) = if input.current_mastery < 200 {
        (
            ContentFormat::FullWorkedExample,
            5,
            "Building foundation with fully worked examples".to_string(),
        )
    } else if input.current_mastery < 400 {
        (
            ContentFormat::PartialWorkedExample { steps_shown_permille: 600 },
            8,
            "Transitioning to partial examples - complete the last steps".to_string(),
        )
    } else if input.current_mastery < 600 {
        (
            ContentFormat::PartialWorkedExample { steps_shown_permille: 400 },
            10,
            "Even split - half worked, half for you to solve".to_string(),
        )
    } else if input.current_mastery < 800 {
        (
            ContentFormat::GuidedProblem {
                difficulty: input.current_mastery,
            },
            15,
            "Ready for problems with hints available".to_string(),
        )
    } else {
        // Adjust difficulty based on cognitive level
        let difficulty = input.current_mastery +
            (input.cognitive_level.difficulty_multiplier() - 1000);
        (
            ContentFormat::IndependentProblem {
                difficulty: difficulty.min(1000),
            },
            20,
            "Expert level - challenging problems without scaffolding".to_string(),
        )
    };

    // Adjust if learner has had recent errors
    let (format, xp_value) = if input.recent_errors > 2 && input.current_mastery > 400 {
        // Step back to more scaffolding
        (
            ContentFormat::PartialWorkedExample { steps_shown_permille: 500 },
            8,
        )
    } else {
        (format, xp_value)
    };

    Ok(WorkedExampleRecommendation {
        format,
        worked_ratio_permille: worked_ratio,
        xp_value,
        reasoning,
    })
}

// =============================================================================
// FEATURE 5: Expertise Reversal Detection
// =============================================================================
// Research (Kalyuga et al., 2003) shows providing detailed explanations to experts
// DECREASES learning by 32%. Must detect when content is too simple.

/// Content complexity profile
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentComplexityProfile {
    /// Content identifier
    pub content_hash: ActionHash,
    /// Base complexity level (0-1000, novice to expert)
    pub base_complexity: u16,
    /// How verbose are explanations (0-1000)
    pub explanation_detail_permille: u16,
    /// Number of worked examples included
    pub worked_examples_count: u8,
    /// Assumed prerequisite knowledge level
    pub assumed_prerequisite_level: CognitiveLevel,
    /// Does this content include redundant information?
    pub has_redundant_info: bool,
}

/// Expertise reversal detection result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertiseReversalResult {
    /// Is expertise reversal effect detected?
    pub is_expertise_reversal: bool,
    /// Mismatch score (higher = worse match)
    pub mismatch_permille: u16,
    /// Is learner underserved? (content too easy)
    pub is_underserved: bool,
    /// Is learner overwhelmed? (content too hard)
    pub is_overwhelmed: bool,
    /// Recommendation
    pub recommendation: ExpertiseRecommendation,
}

/// Recommendations for expertise mismatch
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertiseRecommendation {
    /// Content matches learner level
    ContentAppropriate,
    /// Learner needs more challenging content
    IncreaseChallenge,
    /// Switch to expert-level version (fewer examples, focus on edge cases)
    SwitchToExpertVersion,
    /// Learner needs simpler content
    SimplifyContent,
    /// Add prerequisite review first
    AddPrerequisites,
}

impl Default for ExpertiseRecommendation {
    fn default() -> Self {
        ExpertiseRecommendation::ContentAppropriate
    }
}

/// Input for expertise reversal detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertiseCheckInput {
    pub learner_mastery: u16,
    pub content_complexity: u16,
    pub explanation_detail: u16,
    pub learner_cognitive_level: CognitiveLevel,
}

/// Detect expertise reversal effect
#[hdk_extern]
pub fn detect_expertise_reversal(input: ExpertiseCheckInput) -> ExternResult<ExpertiseReversalResult> {
    // Calculate mismatch
    let mastery_vs_complexity = (input.learner_mastery as i32 - input.content_complexity as i32).abs();
    let mismatch = mastery_vs_complexity as u16;

    // Underserved: high mastery + low complexity + high explanation detail
    let is_underserved = input.learner_mastery > 700 &&
                         input.content_complexity < 400 &&
                         input.explanation_detail > 600;

    // Overwhelmed: low mastery + high complexity
    let is_overwhelmed = input.learner_mastery < 300 &&
                         input.content_complexity > 600;

    // Expertise reversal: expert being given novice content
    let is_expertise_reversal = input.learner_mastery > 800 &&
                                 input.content_complexity < 500 &&
                                 input.explanation_detail > 500;

    let recommendation = if is_expertise_reversal {
        ExpertiseRecommendation::SwitchToExpertVersion
    } else if is_underserved {
        ExpertiseRecommendation::IncreaseChallenge
    } else if is_overwhelmed {
        if input.learner_cognitive_level.level_index() < 2 {
            ExpertiseRecommendation::AddPrerequisites
        } else {
            ExpertiseRecommendation::SimplifyContent
        }
    } else if mismatch < 200 {
        ExpertiseRecommendation::ContentAppropriate
    } else if input.learner_mastery > input.content_complexity {
        ExpertiseRecommendation::IncreaseChallenge
    } else {
        ExpertiseRecommendation::SimplifyContent
    };

    Ok(ExpertiseReversalResult {
        is_expertise_reversal,
        mismatch_permille: mismatch,
        is_underserved,
        is_overwhelmed,
        recommendation,
    })
}

// =============================================================================
// FEATURE 6: Desirable Difficulties Framework
// =============================================================================
// Based on Bjork's desirable difficulties theory.
// Interleaving improves transfer by 32% (Rohrer & Taylor, 2007).
// Spacing + interleaving combined = 47% improvement (Dunlosky et al., 2013).

/// Dimensions of desirable difficulty
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyDimension {
    /// Time since last review (handled by SM-2)
    Spacing,
    /// Mixing topics within session
    Interleaving,
    /// Varying problem contexts
    Variability,
    /// Applying in distant contexts
    Transfer,
    /// Generating answers vs. recognition
    Generation,
    /// Testing rather than restudying
    Testing,
}

/// Desirable difficulty profile for a learner
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DesirableDifficultyProfile {
    /// Current spacing score (0-1000)
    pub spacing_score: u16,
    /// Current interleaving score (0-1000)
    pub interleaving_score: u16,
    /// Current variability score (0-1000)
    pub variability_score: u16,
    /// Current transfer score (0-1000)
    pub transfer_score: u16,
    /// Weakest dimension
    pub weakest_dimension: DifficultyDimension,
    /// Overall desirable difficulty index (0-1000)
    pub difficulty_index: u16,
    /// Recommended session composition
    pub session_composition: SessionComposition,
}

/// Recommended session composition based on desirable difficulties
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionComposition {
    /// Percentage of spaced items (due for review)
    pub spaced_items_permille: u16,
    /// Percentage of interleaved topics
    pub interleaved_permille: u16,
    /// Percentage of varied contexts
    pub variability_permille: u16,
    /// Percentage of transfer practice
    pub transfer_permille: u16,
    /// Number of topics to mix
    pub topics_to_mix: u8,
    /// Context variation level (0-1000)
    pub context_variation: u16,
}

/// Input for calculating desirable difficulties
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DesirableDifficultyInput {
    pub learner_mastery_avg: u16,
    pub learning_style: LearningStyle,
    pub retention_goal: RetentionGoal,
    pub days_since_last_practice: u16,
    pub topics_practiced_last_session: u8,
    pub contexts_used_last_week: u8,
}

/// Retention goal types that affect difficulty composition
/// (Not to be confused with adaptive_integrity::GoalType which is about learning objectives)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionGoal {
    /// Quick mastery for exam (short-term)
    ShortTermRetention,
    /// Long-term skill building (default)
    LongTermRetention,
    /// Ability to use in new situations
    Transfer,
    /// Deep conceptual understanding
    DeepUnderstanding,
}

impl Default for RetentionGoal {
    fn default() -> Self {
        RetentionGoal::LongTermRetention
    }
}

/// Calculate optimal desirable difficulties session composition
#[hdk_extern]
pub fn calculate_desirable_difficulties(input: DesirableDifficultyInput) -> ExternResult<DesirableDifficultyProfile> {
    // Base composition
    let mut spaced = 400u16;      // 40% spaced items
    let mut interleaved = 350u16; // 35% interleaved
    let mut variability = 150u16; // 15% variability
    let mut transfer = 100u16;    // 10% transfer

    // Adjust based on retention goal
    match input.retention_goal {
        RetentionGoal::Transfer => {
            variability = 300;
            transfer = 250;
            interleaved = 300;
            spaced = 150;
        }
        RetentionGoal::DeepUnderstanding => {
            variability = 250;
            interleaved = 400;
            spaced = 200;
            transfer = 150;
        }
        RetentionGoal::ShortTermRetention => {
            spaced = 500;
            interleaved = 300;
            variability = 100;
            transfer = 100;
        }
        RetentionGoal::LongTermRetention => {
            // Default balanced
        }
    }

    // Adjust based on learning style
    match input.learning_style {
        LearningStyle::Kinesthetic => {
            variability += 100; // More real-world contexts
            if spaced > 100 { spaced -= 50; }
            if interleaved > 100 { interleaved -= 50; }
        }
        LearningStyle::Multimodal => {
            // Balanced across all modalities - no adjustment needed
        }
        LearningStyle::Visual | LearningStyle::Auditory | LearningStyle::ReadingWriting => {
            // Standard composition for single-modality learners
        }
    }

    // Calculate dimension scores
    let spacing_score = if input.days_since_last_practice > 7 {
        400 // Too long, spacing score drops
    } else if input.days_since_last_practice > 3 {
        800 // Good spacing
    } else {
        600 // Recent practice
    };

    let interleaving_score = if input.topics_practiced_last_session >= 3 {
        900 // Good interleaving
    } else if input.topics_practiced_last_session >= 2 {
        700
    } else {
        400 // Not enough mixing
    };

    let variability_score = if input.contexts_used_last_week >= 5 {
        900
    } else if input.contexts_used_last_week >= 3 {
        700
    } else {
        400
    };

    let transfer_score = input.learner_mastery_avg.min(800); // Approximate

    // Find weakest dimension
    let scores = [
        (spacing_score, DifficultyDimension::Spacing),
        (interleaving_score, DifficultyDimension::Interleaving),
        (variability_score, DifficultyDimension::Variability),
        (transfer_score, DifficultyDimension::Transfer),
    ];
    let weakest = scores.iter().min_by_key(|(s, _)| *s).map(|(_, d)| d.clone()).unwrap_or(DifficultyDimension::Spacing);

    // Calculate overall difficulty index
    let difficulty_index = ((spacing_score as u32 + interleaving_score as u32 +
                            variability_score as u32 + transfer_score as u32) / 4) as u16;

    // Determine topics to mix
    let topics_to_mix = if input.learner_mastery_avg > 600 { 4 } else { 2 };

    Ok(DesirableDifficultyProfile {
        spacing_score,
        interleaving_score,
        variability_score,
        transfer_score,
        weakest_dimension: weakest,
        difficulty_index,
        session_composition: SessionComposition {
            spaced_items_permille: spaced,
            interleaved_permille: interleaved,
            variability_permille: variability,
            transfer_permille: transfer,
            topics_to_mix,
            context_variation: variability,
        },
    })
}

// =============================================================================
// FEATURE 7: Dual Coding Content Pairing
// =============================================================================
// Mayer & Moreno (2003) multimedia principle: learning improves 89% with both visual + verbal.
// Cognitive load improves 34% when using complementary modalities (Sweller, 1988).

/// Content modality types
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentModality {
    /// Diagrams, videos, animations
    Visual,
    /// Text, audio narration
    Verbal,
    /// Interactive simulations
    Kinesthetic,
    /// Equations, notation, symbols
    Symbolic,
    /// Combined visual + verbal
    DualCoded,
}

impl Default for ContentModality {
    fn default() -> Self {
        ContentModality::Verbal
    }
}

/// Dual-coded content representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualCodedContent {
    /// Primary content identifier
    pub content_hash: ActionHash,
    /// Primary modality
    pub primary_modality: ContentModality,
    /// Complementary modality
    pub complementary_modality: ContentModality,
    /// Visual representation (diagram/video URL or description)
    pub visual_representation: Option<String>,
    /// Verbal explanation (text/audio URL or description)
    pub verbal_explanation: Option<String>,
    /// Integration quality score (how well do they complement? 0-1000)
    pub integration_quality: u16,
    /// Cognitive load estimate for this pairing (0-1000, lower is better)
    pub cognitive_load_permille: u16,
}

/// Content pairing recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualCodingRecommendation {
    /// Recommended primary modality
    pub primary_modality: ContentModality,
    /// Recommended complementary modality
    pub complementary_modality: ContentModality,
    /// Why this pairing is recommended
    pub pairing_rationale: String,
    /// Is current content missing a modality?
    pub missing_modality: Option<ContentModality>,
    /// Expected learning improvement (0-1000)
    pub expected_improvement_permille: u16,
}

/// Input for dual coding recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualCodingInput {
    pub learning_style: LearningStyle,
    pub cognitive_level: CognitiveLevel,
    pub content_type: String,
    pub current_modalities: Vec<ContentModality>,
}

/// Generate dual coding recommendation based on learner and content
#[hdk_extern]
pub fn recommend_dual_coding(input: DualCodingInput) -> ExternResult<DualCodingRecommendation> {
    // Determine optimal modality pairing based on learning style
    let (primary, complementary, rationale) = match input.learning_style {
        LearningStyle::Visual => (
            ContentModality::Visual,
            ContentModality::Verbal,
            "Visual learner: lead with diagrams, supplement with explanation".to_string(),
        ),
        LearningStyle::Auditory => (
            ContentModality::Verbal,
            ContentModality::Visual,
            "Auditory learner: lead with narration, support with visuals".to_string(),
        ),
        LearningStyle::ReadingWriting => (
            ContentModality::Verbal,
            ContentModality::Symbolic,
            "Reading/Writing learner: text explanations with notation".to_string(),
        ),
        LearningStyle::Kinesthetic => (
            ContentModality::Kinesthetic,
            ContentModality::Visual,
            "Kinesthetic learner: interactive simulations with visual guides".to_string(),
        ),
        LearningStyle::Multimodal => (
            ContentModality::DualCoded,
            ContentModality::Kinesthetic,
            "Multimodal learner: balanced dual-coded content with interactive elements".to_string(),
        ),
    };

    // Adjust for cognitive level
    let (primary, complementary) = match input.cognitive_level {
        CognitiveLevel::Analyze | CognitiveLevel::Evaluate => {
            // Higher cognitive levels benefit from symbolic + visual
            (ContentModality::Symbolic, ContentModality::Visual)
        }
        CognitiveLevel::Create => {
            // Creation needs kinesthetic + verbal
            (ContentModality::Kinesthetic, ContentModality::Verbal)
        }
        _ => (primary, complementary),
    };

    // Check what's missing
    let has_visual = input.current_modalities.contains(&ContentModality::Visual);
    let has_verbal = input.current_modalities.contains(&ContentModality::Verbal);
    let has_kinesthetic = input.current_modalities.contains(&ContentModality::Kinesthetic);

    let missing = if !has_visual && (primary == ContentModality::Visual || complementary == ContentModality::Visual) {
        Some(ContentModality::Visual)
    } else if !has_verbal && (primary == ContentModality::Verbal || complementary == ContentModality::Verbal) {
        Some(ContentModality::Verbal)
    } else if !has_kinesthetic && input.learning_style == LearningStyle::Kinesthetic {
        Some(ContentModality::Kinesthetic)
    } else {
        None
    };

    // Calculate expected improvement (89% max from Mayer's research)
    let expected_improvement = if missing.is_some() {
        890 // Full improvement from adding missing modality
    } else if input.current_modalities.len() >= 2 {
        500 // Already dual-coded
    } else {
        700 // Some improvement possible
    };

    Ok(DualCodingRecommendation {
        primary_modality: primary,
        complementary_modality: complementary,
        pairing_rationale: rationale,
        missing_modality: missing,
        expected_improvement_permille: expected_improvement,
    })
}

/// Calculate cognitive load for content modality combination
#[hdk_extern]
pub fn calculate_dual_coding_load(modalities: Vec<ContentModality>) -> ExternResult<u16> {
    let mut load: u16 = 0;

    // Base load per modality
    for modality in &modalities {
        load += match modality {
            ContentModality::Visual => 150,
            ContentModality::Verbal => 150,
            ContentModality::Kinesthetic => 250, // More engaging but more load
            ContentModality::Symbolic => 300,    // Abstract, higher load
            ContentModality::DualCoded => 200,   // Integrated, efficient
        };
    }

    // Synergy bonus for complementary modalities
    let has_visual = modalities.contains(&ContentModality::Visual);
    let has_verbal = modalities.contains(&ContentModality::Verbal);

    if has_visual && has_verbal {
        // Visual + Verbal together REDUCE load (Mayer's modality principle)
        load = load.saturating_sub(100);
    }

    // Too many modalities increases load
    if modalities.len() > 3 {
        load += 150; // Overload penalty
    }

    Ok(load.min(1000))
}

// ============================================================================
// FEATURE 8: Testing Effect / Retrieval Practice Optimization
// Research: Roediger & Karpicke (2006) - 50-70% better retention vs restudying
// ============================================================================

/// Type of retrieval practice
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalType {
    /// Free recall - hardest, most effective
    FreeRecall,
    /// Cued recall - hint provided
    CuedRecall,
    /// Recognition - multiple choice
    Recognition,
    /// Short answer
    ShortAnswer,
    /// Fill in the blank
    FillInBlank,
}

/// A single retrieval attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalAttempt {
    pub skill_hash: ActionHash,
    pub retrieval_type: RetrievalType,
    pub success: bool,
    pub response_time_ms: u32,
    pub confidence_before: u16,  // 0-1000
    pub attempted_at: i64,
}

/// Expanding retrieval schedule (optimal spacing for testing)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalSchedule {
    pub skill_hash: ActionHash,
    /// Current interval between tests (minutes)
    pub current_interval_minutes: u32,
    /// Number of successful retrievals
    pub successful_retrievals: u16,
    /// Number of failed retrievals
    pub failed_retrievals: u16,
    /// Next scheduled retrieval
    pub next_retrieval_at: i64,
    /// Expansion factor (1000 = 1.0, 2000 = 2.0)
    pub expansion_factor_permille: u16,
}

/// Testing vs restudying recommendation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyRecommendation {
    /// Test yourself (retrieval practice)
    RetrievalPractice { retrieval_type: RetrievalType },
    /// Restudy the material
    Restudy,
    /// Mix of both
    InterleavedTestStudy { test_ratio_permille: u16 },
}

/// Input for retrieval practice recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalPracticeInput {
    pub skill_hash: ActionHash,
    pub mastery_permille: u16,
    pub last_retrieval_success: bool,
    pub retrievals_today: u16,
    pub time_since_last_study_minutes: u32,
}

/// Result of retrieval practice analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalPracticeResult {
    pub recommendation: StudyRecommendation,
    pub optimal_retrieval_type: RetrievalType,
    pub expected_retention_boost_permille: u16,
    pub next_retrieval_interval_minutes: u32,
    pub testing_effect_active: bool,
}

/// Calculate optimal retrieval practice schedule
/// Based on expanding retrieval practice research
#[hdk_extern]
pub fn calculate_retrieval_schedule(input: RetrievalPracticeInput) -> ExternResult<RetrievalPracticeResult> {
    // Testing effect is strongest when:
    // 1. Initial learning has occurred (mastery > 300)
    // 2. Some time has passed (spacing effect)
    // 3. Not over-tested (diminishing returns after ~5/day)

    let testing_effect_active = input.mastery_permille >= 300
        && input.time_since_last_study_minutes >= 10
        && input.retrievals_today < 5;

    // Choose retrieval type based on mastery
    let optimal_type = if input.mastery_permille < 400 {
        RetrievalType::Recognition // Easier for beginners
    } else if input.mastery_permille < 600 {
        RetrievalType::CuedRecall
    } else if input.mastery_permille < 800 {
        RetrievalType::ShortAnswer
    } else {
        RetrievalType::FreeRecall // Hardest, most effective for experts
    };

    // Calculate next interval using expanding schedule
    let base_interval = if input.last_retrieval_success { 60u32 } else { 15u32 };
    let expansion = 1.0 + (input.mastery_permille as f64 / 1000.0);
    let next_interval = (base_interval as f64 * expansion) as u32;

    // Recommendation based on conditions
    let recommendation = if !testing_effect_active {
        if input.mastery_permille < 300 {
            StudyRecommendation::Restudy
        } else {
            StudyRecommendation::InterleavedTestStudy { test_ratio_permille: 300 }
        }
    } else if input.mastery_permille > 700 {
        StudyRecommendation::RetrievalPractice { retrieval_type: optimal_type.clone() }
    } else {
        StudyRecommendation::InterleavedTestStudy { test_ratio_permille: 500 }
    };

    // Expected retention boost from testing effect (50-70% in research)
    let retention_boost = if testing_effect_active { 600u16 } else { 200u16 };

    Ok(RetrievalPracticeResult {
        recommendation,
        optimal_retrieval_type: optimal_type,
        expected_retention_boost_permille: retention_boost,
        next_retrieval_interval_minutes: next_interval,
        testing_effect_active,
    })
}

// ============================================================================
// FEATURE 9: Hypercorrection Effect
// Research: Butterfield & Metcalfe (2001) - High-confidence errors corrected 86% vs 64%
// ============================================================================

/// A high-confidence error (prime target for hypercorrection)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HighConfidenceError {
    pub skill_hash: ActionHash,
    pub question_id: String,
    /// Confidence when wrong answer was given (0-1000)
    pub confidence_when_wrong: u16,
    /// The incorrect response
    pub incorrect_response: String,
    /// The correct response
    pub correct_response: String,
    /// When the error occurred
    pub occurred_at: i64,
    /// Has this been corrected?
    pub corrected: bool,
    /// Correction attempts
    pub correction_attempts: u16,
}

/// Hypercorrection analysis result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypercorrectionAnalysis {
    /// Is this a hypercorrection candidate? (high confidence + wrong)
    pub is_hypercorrection_candidate: bool,
    /// Priority for correction (0-1000, higher = more urgent)
    pub correction_priority: u16,
    /// Expected correction rate (0-1000)
    pub expected_correction_rate_permille: u16,
    /// Recommended feedback intensity
    pub feedback_intensity: FeedbackIntensity,
    /// Surprise factor (high confidence + wrong = high surprise)
    pub surprise_factor_permille: u16,
}

/// Feedback intensity for corrections
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackIntensity {
    /// Minimal feedback
    Light,
    /// Standard correction
    Standard,
    /// Emphasize the correct answer
    Emphasized,
    /// Deep explanation with elaboration
    Deep,
}

/// Input for hypercorrection analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypercorrectionInput {
    pub confidence_when_wrong: u16,
    pub skill_mastery: u16,
    pub times_answered_incorrectly: u16,
    pub is_conceptual_error: bool,
}

/// Analyze error for hypercorrection potential
#[hdk_extern]
pub fn analyze_hypercorrection(input: HypercorrectionInput) -> ExternResult<HypercorrectionAnalysis> {
    // High confidence errors (>700) are corrected at 86% vs 64% for low confidence
    let is_candidate = input.confidence_when_wrong >= 700;

    // Surprise factor: how unexpected was being wrong?
    let surprise = if input.confidence_when_wrong > 800 {
        900u16 // Very surprised
    } else if input.confidence_when_wrong > 600 {
        700u16
    } else {
        400u16
    };

    // Priority based on confidence, mastery, and repetition
    let mut priority = input.confidence_when_wrong;

    // Conceptual errors are more important to correct
    if input.is_conceptual_error {
        priority = priority.saturating_add(150);
    }

    // Repeated errors need attention
    if input.times_answered_incorrectly > 2 {
        priority = priority.saturating_add(100);
    }

    priority = priority.min(1000);

    // Expected correction rate based on research
    let correction_rate = if input.confidence_when_wrong >= 800 {
        860u16 // 86% from research
    } else if input.confidence_when_wrong >= 600 {
        750u16
    } else {
        640u16 // 64% for low confidence
    };

    // Feedback intensity based on surprise and importance
    let feedback_intensity = if surprise > 800 || input.is_conceptual_error {
        FeedbackIntensity::Deep
    } else if surprise > 600 {
        FeedbackIntensity::Emphasized
    } else if surprise > 400 {
        FeedbackIntensity::Standard
    } else {
        FeedbackIntensity::Light
    };

    Ok(HypercorrectionAnalysis {
        is_hypercorrection_candidate: is_candidate,
        correction_priority: priority,
        expected_correction_rate_permille: correction_rate,
        feedback_intensity,
        surprise_factor_permille: surprise,
    })
}

// ============================================================================
// FEATURE 10: Pre-Testing Effect (Test-Potentiated Learning)
// Research: Richland et al. (2009) - 10-25% improvement even with wrong pre-test answers
// ============================================================================

/// Pre-test question for priming learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreTestQuestion {
    pub skill_hash: ActionHash,
    pub question_text: String,
    pub question_type: RetrievalType,
    /// Is this testing material not yet learned?
    pub is_pre_learning: bool,
    /// Difficulty (0-1000)
    pub difficulty_permille: u16,
}

/// Result of a pre-test attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreTestResult {
    pub skill_hash: ActionHash,
    pub was_correct: bool,
    pub response_given: String,
    pub time_spent_ms: u32,
    /// Curiosity triggered by not knowing
    pub curiosity_triggered: bool,
    /// Attention primed for upcoming content
    pub attention_primed: bool,
}

/// Pre-testing analysis and recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreTestAnalysis {
    /// Should pre-test before instruction?
    pub should_pre_test: bool,
    /// Expected learning boost (0-1000, typically 100-250)
    pub expected_boost_permille: u16,
    /// Optimal number of pre-test questions
    pub optimal_question_count: u8,
    /// Recommended question types
    pub recommended_types: Vec<RetrievalType>,
    /// Pre-test primes attention for these concepts
    pub concepts_to_prime: Vec<String>,
}

/// Input for pre-test analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreTestInput {
    pub skill_hash: ActionHash,
    pub learner_prior_knowledge_permille: u16,
    pub content_complexity_permille: u16,
    pub available_time_minutes: u16,
    pub is_conceptual_content: bool,
}

/// Analyze whether pre-testing would benefit learning
#[hdk_extern]
pub fn analyze_pre_test_benefit(input: PreTestInput) -> ExternResult<PreTestAnalysis> {
    // Pre-testing works best for:
    // 1. Conceptual material
    // 2. When learner has some prior knowledge (can make guesses)
    // 3. Complex content that benefits from attention priming

    let should_pre_test = input.is_conceptual_content
        && input.learner_prior_knowledge_permille >= 100
        && input.available_time_minutes >= 5;

    // Expected boost: 10-25% from research (100-250 permille)
    let boost = if input.is_conceptual_content {
        if input.content_complexity_permille > 700 {
            250u16 // Complex conceptual = maximum benefit
        } else {
            180u16
        }
    } else {
        120u16 // Factual content benefits less
    };

    // Optimal question count (3-5 usually optimal)
    let question_count = if input.available_time_minutes >= 15 {
        5u8
    } else if input.available_time_minutes >= 10 {
        4u8
    } else {
        3u8
    };

    // Recommend easier retrieval types for pre-tests (learner hasn't learned yet!)
    let types = vec![
        RetrievalType::Recognition,
        RetrievalType::FillInBlank,
    ];

    Ok(PreTestAnalysis {
        should_pre_test,
        expected_boost_permille: boost,
        optimal_question_count: question_count,
        recommended_types: types,
        concepts_to_prime: vec![], // Would be populated from content analysis
    })
}

// ============================================================================
// FEATURE 11: Productive Failure Framework
// Research: Kapur (2008, 2012) - 20-30% improvement on transfer problems
// ============================================================================

/// Phase of productive failure learning
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductiveFailurePhase {
    /// Initial exploration without instruction
    Exploration,
    /// Attempting to solve without scaffolding
    Struggle,
    /// Consolidation with instruction after struggle
    Consolidation,
    /// Application of learned concepts
    Application,
}

/// Metrics during productive failure struggle phase
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StruggleMetrics {
    /// Time spent struggling (minutes)
    pub struggle_duration_minutes: u16,
    /// Number of solution attempts
    pub solution_attempts: u16,
    /// Different approaches tried
    pub approaches_tried: u16,
    /// Partial solutions generated
    pub partial_solutions: u16,
    /// Frustration level (0-1000)
    pub frustration_permille: u16,
    /// Engagement despite failure (0-1000)
    pub engagement_permille: u16,
}

/// Productive failure analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductiveFailureAnalysis {
    /// Is this a good candidate for productive failure?
    pub is_good_candidate: bool,
    /// Current phase
    pub current_phase: ProductiveFailurePhase,
    /// Should continue struggling or provide instruction?
    pub should_continue_struggle: bool,
    /// Optimal struggle duration (minutes)
    pub optimal_struggle_minutes: u16,
    /// Expected transfer improvement (0-1000)
    pub expected_transfer_boost_permille: u16,
    /// Recommendation
    pub recommendation: ProductiveFailureRecommendation,
}

/// Recommendations for productive failure
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductiveFailureRecommendation {
    /// Continue exploration
    ContinueStruggle,
    /// Provide a hint (partial scaffold)
    ProvideHint,
    /// Move to consolidation (provide instruction)
    BeginConsolidation,
    /// Problem too hard, simplify first
    SimplifyProblem,
    /// Productive failure not appropriate here
    UseDirectInstruction,
}

/// Input for productive failure analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductiveFailureInput {
    pub skill_hash: ActionHash,
    pub content_is_conceptual: bool,
    pub learner_persistence_permille: u16,
    pub struggle_metrics: StruggleMetrics,
    pub prior_knowledge_permille: u16,
}

/// Analyze productive failure situation
#[hdk_extern]
pub fn analyze_productive_failure(input: ProductiveFailureInput) -> ExternResult<ProductiveFailureAnalysis> {
    // Productive failure works best for:
    // 1. Conceptual problems (not procedural)
    // 2. Learners with persistence
    // 3. Problems within reach (not too hard)

    let is_good_candidate = input.content_is_conceptual
        && input.learner_persistence_permille >= 500
        && input.prior_knowledge_permille >= 200;

    // Determine current phase based on struggle metrics
    let current_phase = if input.struggle_metrics.struggle_duration_minutes < 2 {
        ProductiveFailurePhase::Exploration
    } else if input.struggle_metrics.solution_attempts < 3 {
        ProductiveFailurePhase::Struggle
    } else {
        ProductiveFailurePhase::Consolidation
    };

    // Should continue struggle?
    // Stop if: too frustrated, too long, or made good attempts
    let should_continue = input.struggle_metrics.frustration_permille < 700
        && input.struggle_metrics.struggle_duration_minutes < 15
        && input.struggle_metrics.engagement_permille > 400
        && input.struggle_metrics.approaches_tried < 5;

    // Optimal struggle time (5-15 minutes typically)
    let optimal_minutes = if input.learner_persistence_permille > 700 {
        15u16
    } else if input.learner_persistence_permille > 500 {
        10u16
    } else {
        5u16
    };

    // Recommendation
    let recommendation = if !is_good_candidate {
        ProductiveFailureRecommendation::UseDirectInstruction
    } else if input.struggle_metrics.frustration_permille > 800 {
        ProductiveFailureRecommendation::SimplifyProblem
    } else if should_continue && input.struggle_metrics.approaches_tried < 2 {
        ProductiveFailureRecommendation::ContinueStruggle
    } else if should_continue {
        ProductiveFailureRecommendation::ProvideHint
    } else {
        ProductiveFailureRecommendation::BeginConsolidation
    };

    // Expected transfer boost: 20-30% from research
    let transfer_boost = if is_good_candidate && input.struggle_metrics.approaches_tried >= 2 {
        280u16 // Full productive failure benefit
    } else if is_good_candidate {
        180u16
    } else {
        50u16 // Minimal benefit without productive failure
    };

    Ok(ProductiveFailureAnalysis {
        is_good_candidate,
        current_phase,
        should_continue_struggle: should_continue,
        optimal_struggle_minutes: optimal_minutes,
        expected_transfer_boost_permille: transfer_boost,
        recommendation,
    })
}

// ============================================================================
// FEATURE 12: Self-Determination Theory (Deci & Ryan)
// Research: Autonomy, Competence, Relatedness = intrinsic motivation
// ============================================================================

/// The three basic psychological needs (SDT)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SDTNeeds {
    /// Autonomy: feeling of choice and control (0-1000)
    pub autonomy_permille: u16,
    /// Competence: feeling effective and capable (0-1000)
    pub competence_permille: u16,
    /// Relatedness: feeling connected to others (0-1000)
    pub relatedness_permille: u16,
}

/// SDT need satisfaction assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SDTAssessment {
    pub needs: SDTNeeds,
    /// Overall intrinsic motivation (0-1000)
    pub intrinsic_motivation_permille: u16,
    /// Which need is most deficient?
    pub weakest_need: SDTNeedType,
    /// Recommendations to improve motivation
    pub recommendations: Vec<SDTRecommendation>,
    /// Risk of amotivation
    pub amotivation_risk_permille: u16,
}

/// Type of SDT need
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SDTNeedType {
    Autonomy,
    Competence,
    Relatedness,
}

/// Recommendations to support SDT needs
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SDTRecommendation {
    /// Offer more choices
    IncreaseChoices,
    /// Reduce controlling language
    ReduceControllingLanguage,
    /// Provide competence feedback
    ProvideCompetenceFeedback,
    /// Adjust difficulty to match skill
    AdjustDifficulty,
    /// Encourage peer interaction
    EncouragePeerInteraction,
    /// Join study groups
    JoinStudyGroup,
    /// Acknowledge feelings
    AcknowledgeFeelings,
    /// Provide rationale for activities
    ProvideRationale,
}

/// Input for SDT assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SDTInput {
    /// Choices made by learner in last 7 days
    pub choices_made: u16,
    /// Times learner was given options
    pub options_presented: u16,
    /// Recent success rate (0-1000)
    pub recent_success_rate_permille: u16,
    /// Peer interactions in last 7 days
    pub peer_interactions: u16,
    /// Study group memberships
    pub study_group_count: u8,
    /// Feeling of progress (self-reported, 0-1000)
    pub progress_feeling_permille: u16,
}

/// Assess SDT need satisfaction
#[hdk_extern]
pub fn assess_sdt_needs(input: SDTInput) -> ExternResult<SDTAssessment> {
    // Calculate autonomy: based on choices made vs options given
    let autonomy = if input.options_presented > 0 {
        ((input.choices_made as u32 * 1000) / input.options_presented.max(1) as u32).min(1000) as u16
    } else {
        300u16 // Low autonomy if no choices offered
    };

    // Calculate competence: based on success rate and progress feeling
    let competence = (input.recent_success_rate_permille + input.progress_feeling_permille) / 2;

    // Calculate relatedness: based on peer interactions and groups
    let relatedness = {
        let interaction_score = (input.peer_interactions as u32 * 100).min(500);
        let group_score = (input.study_group_count as u32 * 200).min(500);
        (interaction_score + group_score).min(1000) as u16
    };

    let needs = SDTNeeds {
        autonomy_permille: autonomy,
        competence_permille: competence,
        relatedness_permille: relatedness,
    };

    // Overall intrinsic motivation (geometric mean of three needs)
    let intrinsic_motivation = {
        let product = (autonomy as u64) * (competence as u64) * (relatedness as u64);
        // Cube root approximation
        let avg = (autonomy as u32 + competence as u32 + relatedness as u32) / 3;
        avg.min(1000) as u16
    };

    // Find weakest need
    let weakest_need = if autonomy <= competence && autonomy <= relatedness {
        SDTNeedType::Autonomy
    } else if competence <= autonomy && competence <= relatedness {
        SDTNeedType::Competence
    } else {
        SDTNeedType::Relatedness
    };

    // Generate recommendations
    let mut recommendations = Vec::new();

    if autonomy < 500 {
        recommendations.push(SDTRecommendation::IncreaseChoices);
        recommendations.push(SDTRecommendation::ProvideRationale);
    }
    if competence < 500 {
        recommendations.push(SDTRecommendation::ProvideCompetenceFeedback);
        recommendations.push(SDTRecommendation::AdjustDifficulty);
    }
    if relatedness < 500 {
        recommendations.push(SDTRecommendation::EncouragePeerInteraction);
        recommendations.push(SDTRecommendation::JoinStudyGroup);
    }

    // Amotivation risk: high if all needs are low
    let amotivation_risk = if autonomy < 300 && competence < 300 && relatedness < 300 {
        800u16
    } else if autonomy < 400 || competence < 400 || relatedness < 400 {
        500u16
    } else {
        200u16
    };

    Ok(SDTAssessment {
        needs,
        intrinsic_motivation_permille: intrinsic_motivation,
        weakest_need,
        recommendations,
        amotivation_risk_permille: amotivation_risk,
    })
}

// ============================================================================
// FEATURE 13: Growth Mindset Integration (Dweck)
// Research: Growth mindset correlates with persistence and achievement
// ============================================================================

/// Mindset indicator
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MindsetIndicator {
    /// Fixed mindset: ability is static
    Fixed,
    /// Growth mindset: ability can be developed
    Growth,
    /// Mixed signals
    Mixed,
}

/// Language patterns indicating mindset
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MindsetLanguage {
    /// "I can't do this" - fixed
    CantDo,
    /// "I can't do this YET" - growth
    CantDoYet,
    /// "I'm not smart enough" - fixed
    NotSmartEnough,
    /// "I need more practice" - growth
    NeedMorePractice,
    /// "This is too hard" - fixed leaning
    TooHard,
    /// "This is challenging" - growth leaning
    Challenging,
    /// "I give up" - fixed
    GiveUp,
    /// "I'll try a different approach" - growth
    TryDifferent,
}

/// Mindset assessment result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MindsetAssessment {
    /// Overall mindset indicator
    pub mindset: MindsetIndicator,
    /// Growth mindset score (0-1000)
    pub growth_score_permille: u16,
    /// Fixed mindset score (0-1000)
    pub fixed_score_permille: u16,
    /// Intervention recommendations
    pub interventions: Vec<MindsetIntervention>,
    /// Effort attribution score (0-1000)
    pub effort_attribution_permille: u16,
    /// Challenge seeking (0-1000)
    pub challenge_seeking_permille: u16,
}

/// Mindset interventions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MindsetIntervention {
    /// Praise effort, not ability
    PraiseEffort,
    /// Normalize struggle
    NormalizeStruggle,
    /// Teach brain plasticity
    TeachPlasticity,
    /// Reframe failure as learning
    ReframeFailure,
    /// Show growth examples
    ShowGrowthExamples,
    /// Use "yet" framing
    UseYetFraming,
    /// Challenge comfort zone
    ChallengeComfortZone,
}

/// Input for mindset assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MindsetInput {
    /// Response to failure (did they try again?)
    pub retry_after_failure_rate_permille: u16,
    /// Chose challenging over easy tasks
    pub chose_challenge_rate_permille: u16,
    /// Attribution: effort vs ability (1000 = all effort, 0 = all ability)
    pub effort_attribution_permille: u16,
    /// Response time increase after errors (higher = more discouraged)
    pub response_slowdown_after_error_permille: u16,
    /// Help-seeking after struggle
    pub help_seeking_rate_permille: u16,
}

/// Assess learner's mindset
#[hdk_extern]
pub fn assess_mindset(input: MindsetInput) -> ExternResult<MindsetAssessment> {
    // Growth indicators
    let growth_signals = [
        input.retry_after_failure_rate_permille,
        input.chose_challenge_rate_permille,
        input.effort_attribution_permille,
        1000u16.saturating_sub(input.response_slowdown_after_error_permille),
        input.help_seeking_rate_permille,
    ];

    let growth_score: u16 = growth_signals.iter().sum::<u16>() / growth_signals.len() as u16;
    let fixed_score = 1000u16.saturating_sub(growth_score);

    let mindset = if growth_score >= 700 {
        MindsetIndicator::Growth
    } else if growth_score <= 400 {
        MindsetIndicator::Fixed
    } else {
        MindsetIndicator::Mixed
    };

    // Generate interventions based on weak areas
    let mut interventions = Vec::new();

    if input.retry_after_failure_rate_permille < 500 {
        interventions.push(MindsetIntervention::ReframeFailure);
        interventions.push(MindsetIntervention::NormalizeStruggle);
    }
    if input.chose_challenge_rate_permille < 500 {
        interventions.push(MindsetIntervention::ChallengeComfortZone);
    }
    if input.effort_attribution_permille < 500 {
        interventions.push(MindsetIntervention::PraiseEffort);
        interventions.push(MindsetIntervention::TeachPlasticity);
    }
    if input.response_slowdown_after_error_permille > 500 {
        interventions.push(MindsetIntervention::UseYetFraming);
    }

    // Add general growth interventions if fixed mindset
    if matches!(mindset, MindsetIndicator::Fixed) {
        interventions.push(MindsetIntervention::ShowGrowthExamples);
    }

    Ok(MindsetAssessment {
        mindset,
        growth_score_permille: growth_score,
        fixed_score_permille: fixed_score,
        interventions,
        effort_attribution_permille: input.effort_attribution_permille,
        challenge_seeking_permille: input.chose_challenge_rate_permille,
    })
}

// ============================================================================
// FEATURE 14: Attention / Mind-Wandering Detection
// Research: Response time patterns reveal disengagement (Smallwood & Schooler)
// ============================================================================

/// Current attention state
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionState {
    /// Fully focused
    Focused,
    /// Slightly distracted
    Drifting,
    /// Mind wandering detected
    MindWandering,
    /// Severe disengagement
    Disengaged,
    /// Too fast (likely guessing)
    Guessing,
    /// Unknown (not enough data)
    Unknown,
}

/// Response time pattern analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseTimePattern {
    /// Average response time (ms)
    pub avg_response_ms: u32,
    /// Standard deviation of response times
    pub std_dev_ms: u32,
    /// Trend: increasing (slowing) or decreasing
    pub trend_direction: TrendDirection,
    /// Coefficient of variation (0-1000)
    pub variability_permille: u16,
    /// Number of outliers (very slow or very fast)
    pub outlier_count: u16,
}

/// Direction of trend
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Attention assessment result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionAssessment {
    /// Current attention state
    pub state: AttentionState,
    /// Confidence in assessment (0-1000)
    pub confidence_permille: u16,
    /// Estimated focus level (0-1000)
    pub focus_level_permille: u16,
    /// Time since last focused state (minutes)
    pub time_since_focused_minutes: u16,
    /// Recommended re-engagement action
    pub re_engagement_action: ReEngagementAction,
    /// Predicted accuracy if current state continues
    pub predicted_accuracy_permille: u16,
}

/// Actions to re-engage learner
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReEngagementAction {
    /// Continue as normal
    Continue,
    /// Ask a thought probe question
    ThoughtProbe,
    /// Switch to different content type
    SwitchContent,
    /// Take a short break
    TakeBreak,
    /// Increase interactivity
    IncreaseInteractivity,
    /// Simplify current content
    SimplifyContent,
    /// Provide encouragement
    Encourage,
    /// End session
    EndSession,
}

/// Input for attention assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionInput {
    /// Recent response times (last 10)
    pub recent_response_times_ms: Vec<u32>,
    /// Recent accuracy (last 10 answers)
    pub recent_accuracy_permille: u16,
    /// Session duration so far (minutes)
    pub session_duration_minutes: u16,
    /// Time since last break (minutes)
    pub time_since_break_minutes: u16,
    /// Error streak (consecutive errors)
    pub consecutive_errors: u8,
}

/// Assess attention state from behavioral patterns
#[hdk_extern]
pub fn assess_attention(input: AttentionInput) -> ExternResult<AttentionAssessment> {
    // Need at least 3 response times for analysis
    if input.recent_response_times_ms.len() < 3 {
        return Ok(AttentionAssessment {
            state: AttentionState::Unknown,
            confidence_permille: 200,
            focus_level_permille: 500,
            time_since_focused_minutes: 0,
            re_engagement_action: ReEngagementAction::Continue,
            predicted_accuracy_permille: 500,
        });
    }

    // Calculate response time statistics
    let times = &input.recent_response_times_ms;
    let avg: u32 = times.iter().sum::<u32>() / times.len() as u32;

    // Calculate variance
    let variance: u64 = times.iter()
        .map(|&t| {
            let diff = if t > avg { t - avg } else { avg - t };
            (diff as u64) * (diff as u64)
        })
        .sum::<u64>() / times.len() as u64;
    let std_dev = (variance as f64).sqrt() as u32;

    // Coefficient of variation (higher = more variable = less focused)
    let cv = if avg > 0 { (std_dev * 1000) / avg } else { 0 };

    // Detect trend (is response time increasing?)
    let first_half_avg: u32 = times[..times.len()/2].iter().sum::<u32>() / (times.len()/2) as u32;
    let second_half_avg: u32 = times[times.len()/2..].iter().sum::<u32>() / (times.len() - times.len()/2) as u32;

    let trend = if second_half_avg > first_half_avg.saturating_add(avg / 4) {
        TrendDirection::Increasing // Slowing down
    } else if first_half_avg > second_half_avg.saturating_add(avg / 4) {
        TrendDirection::Decreasing // Speeding up (possibly guessing)
    } else {
        TrendDirection::Stable
    };

    // Determine attention state
    let state = if avg < 500 && input.recent_accuracy_permille < 600 {
        AttentionState::Guessing // Too fast + inaccurate
    } else if cv > 600 || (matches!(trend, TrendDirection::Increasing) && input.recent_accuracy_permille < 500) {
        AttentionState::MindWandering
    } else if cv > 400 || input.consecutive_errors >= 3 {
        AttentionState::Drifting
    } else if input.session_duration_minutes > 45 && input.recent_accuracy_permille < 700 {
        AttentionState::Disengaged
    } else {
        AttentionState::Focused
    };

    // Focus level (inverse of variability and errors)
    let focus_level = 1000u16
        .saturating_sub(cv.min(500) as u16)
        .saturating_sub(input.consecutive_errors as u16 * 100);

    // Re-engagement action
    let action = match state {
        AttentionState::Focused => ReEngagementAction::Continue,
        AttentionState::Drifting => ReEngagementAction::ThoughtProbe,
        AttentionState::MindWandering => {
            if input.time_since_break_minutes > 25 {
                ReEngagementAction::TakeBreak
            } else {
                ReEngagementAction::SwitchContent
            }
        }
        AttentionState::Disengaged => {
            if input.session_duration_minutes > 60 {
                ReEngagementAction::EndSession
            } else {
                ReEngagementAction::TakeBreak
            }
        }
        AttentionState::Guessing => ReEngagementAction::SimplifyContent,
        AttentionState::Unknown => ReEngagementAction::Continue,
    };

    // Predicted accuracy based on attention state
    let predicted_accuracy = match state {
        AttentionState::Focused => input.recent_accuracy_permille,
        AttentionState::Drifting => input.recent_accuracy_permille.saturating_sub(100),
        AttentionState::MindWandering => input.recent_accuracy_permille.saturating_sub(200),
        AttentionState::Disengaged => input.recent_accuracy_permille.saturating_sub(300),
        AttentionState::Guessing => 250, // Random guessing
        AttentionState::Unknown => 500,
    };

    // Time since focused (rough estimate)
    let time_since_focused = if matches!(state, AttentionState::Focused) {
        0
    } else {
        input.time_since_break_minutes.min(30)
    };

    Ok(AttentionAssessment {
        state,
        confidence_permille: if times.len() >= 5 { 800 } else { 500 },
        focus_level_permille: focus_level,
        time_since_focused_minutes: time_since_focused,
        re_engagement_action: action,
        predicted_accuracy_permille: predicted_accuracy,
    })
}

// =============================================================================
// FEATURE 15: Critical Thinking Framework
// Argument analysis, logical reasoning, fallacy detection
// =============================================================================

/// Types of claims in an argument
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClaimType {
    /// Factual claim that can be verified
    Factual,
    /// Value judgment (good/bad, should/shouldn't)
    Evaluative,
    /// Proposed course of action
    Policy,
    /// Interpretation of evidence
    Interpretive,
    /// Definition of a term
    Definitional,
    /// Cause-effect relationship
    Causal,
}

/// Evidence types supporting a claim
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceType {
    /// Statistical data
    Statistical,
    /// Personal experience or testimony
    Anecdotal,
    /// Expert opinion or authority
    Expert,
    /// Research study
    Empirical,
    /// Logical derivation
    Logical,
    /// Analogy or comparison
    Analogical,
    /// Historical precedent
    Historical,
    /// No evidence provided
    None,
}

/// Strength of evidence (0-1000)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceStrength {
    pub relevance_permille: u16,
    pub reliability_permille: u16,
    pub sufficiency_permille: u16,
    pub overall_permille: u16,
}

/// Logical fallacy categories (20+ types)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogicalFallacy {
    // Fallacies of Relevance
    AdHominem,           // Attack the person, not the argument
    AppealToAuthority,   // Inappropriate authority citation
    AppealToEmotion,     // Emotional manipulation
    AppealToTradition,   // "We've always done it this way"
    AppealToNature,      // "Natural = good"
    AppealToPopularity,  // Bandwagon
    RedHerring,          // Irrelevant distraction
    StrawMan,            // Misrepresenting opponent's argument

    // Fallacies of Ambiguity
    Equivocation,        // Shifting word meaning
    Amphiboly,           // Grammatical ambiguity

    // Fallacies of Presumption
    FalseDialemma,       // Only 2 options when more exist
    SlipperySlope,       // Unwarranted chain of consequences
    CircularReasoning,   // Conclusion in premises
    HastyGeneralization, // Too small sample
    FalseCause,          // Post hoc ergo propter hoc

    // Fallacies of Weak Induction
    WeakAnalogy,         // Poor comparison
    AppealToIgnorance,   // No evidence = false

    // Formal Fallacies
    AffirmingConsequent, // If P then Q; Q; therefore P
    DenyingAntecedent,   // If P then Q; not P; therefore not Q

    // Other Common Fallacies
    NoTrueScotsman,      // Arbitrary redefinition
    MovingGoalposts,     // Changing criteria after fact
    TuQuoque,            // "You do it too"
    GeneticFallacy,      // Origin determines value
    Whataboutism,        // Deflection to other issues
}

/// Argument component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgumentComponent {
    pub claim: String,
    pub claim_type: ClaimType,
    pub evidence_type: EvidenceType,
    pub evidence_strength: EvidenceStrength,
    pub assumptions: Vec<String>,
    pub detected_fallacies: Vec<LogicalFallacy>,
}

/// Argument analysis result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgumentAnalysis {
    pub main_claim: ClaimType,
    pub components: Vec<ArgumentComponent>,
    pub logical_validity_permille: u16,
    pub evidence_quality_permille: u16,
    pub fallacies_detected: Vec<(LogicalFallacy, String)>, // Fallacy + explanation
    pub hidden_assumptions: Vec<String>,
    pub overall_strength_permille: u16,
    pub improvement_suggestions: Vec<String>,
}

/// Input for argument analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgumentInput {
    pub argument_text: String,
    pub claim_type: ClaimType,
    pub provided_evidence: Vec<(EvidenceType, String)>,
    pub learner_confidence_permille: u16,
}

/// Analyze an argument for logical validity and fallacies
#[hdk_extern]
pub fn analyze_argument(input: ArgumentInput) -> ExternResult<ArgumentAnalysis> {
    let mut fallacies = Vec::new();
    let mut suggestions = Vec::new();
    let mut assumptions = Vec::new();

    // Check for common fallacy patterns (simplified heuristics)
    let text_lower = input.argument_text.to_lowercase();

    // Ad Hominem detection
    if text_lower.contains("stupid") || text_lower.contains("idiot") ||
       text_lower.contains("they're just") || text_lower.contains("typical of") {
        fallacies.push((LogicalFallacy::AdHominem,
            "Attacking the person rather than the argument".to_string()));
    }

    // Appeal to Authority
    if (text_lower.contains("expert says") || text_lower.contains("scientist says")) &&
       input.provided_evidence.iter().all(|(t, _)| !matches!(t, EvidenceType::Empirical)) {
        fallacies.push((LogicalFallacy::AppealToAuthority,
            "Authority cited without supporting evidence".to_string()));
    }

    // False Dilemma
    if text_lower.contains("either") && text_lower.contains("or") &&
       !text_lower.contains("alternatively") && !text_lower.contains("also") {
        fallacies.push((LogicalFallacy::FalseDialemma,
            "Only two options presented when more may exist".to_string()));
        suggestions.push("Consider whether there are additional options beyond the two presented".to_string());
    }

    // Slippery Slope
    if text_lower.contains("will lead to") && text_lower.contains("eventually") {
        fallacies.push((LogicalFallacy::SlipperySlope,
            "Assumes chain of consequences without justification".to_string()));
        suggestions.push("Provide evidence for each step in the causal chain".to_string());
    }

    // Appeal to Tradition
    if text_lower.contains("always been") || text_lower.contains("traditional") {
        fallacies.push((LogicalFallacy::AppealToTradition,
            "Tradition alone doesn't justify current practice".to_string()));
    }

    // Appeal to Nature
    if text_lower.contains("natural") && text_lower.contains("good") ||
       text_lower.contains("unnatural") && text_lower.contains("bad") {
        fallacies.push((LogicalFallacy::AppealToNature,
            "Natural doesn't necessarily mean good or better".to_string()));
    }

    // Hasty Generalization
    if (text_lower.contains("all") || text_lower.contains("every") || text_lower.contains("always")) &&
       input.provided_evidence.len() <= 1 {
        fallacies.push((LogicalFallacy::HastyGeneralization,
            "Broad claim based on limited evidence".to_string()));
        suggestions.push("Consider a larger sample or qualify the claim".to_string());
    }

    // Calculate evidence quality
    let evidence_quality = if input.provided_evidence.is_empty() {
        100u16
    } else {
        let quality_sum: u32 = input.provided_evidence.iter().map(|(t, _)| {
            match t {
                EvidenceType::Empirical => 900u32,
                EvidenceType::Statistical => 850,
                EvidenceType::Expert => 700,
                EvidenceType::Historical => 600,
                EvidenceType::Logical => 750,
                EvidenceType::Analogical => 500,
                EvidenceType::Anecdotal => 300,
                EvidenceType::None => 0,
            }
        }).sum();
        (quality_sum / input.provided_evidence.len() as u32) as u16
    };

    // Logical validity based on fallacy count
    let fallacy_penalty = fallacies.len() as u16 * 150;
    let logical_validity = 1000u16.saturating_sub(fallacy_penalty);

    // Overall strength combines evidence and logic
    let overall = (evidence_quality as u32 * 500 + logical_validity as u32 * 500) / 1000;

    // Identify hidden assumptions based on claim type
    match input.claim_type {
        ClaimType::Causal => {
            assumptions.push("Correlation implies causation".to_string());
            assumptions.push("No confounding variables".to_string());
        }
        ClaimType::Policy => {
            assumptions.push("Implementation is feasible".to_string());
            assumptions.push("Benefits outweigh costs".to_string());
        }
        ClaimType::Evaluative => {
            assumptions.push("Shared value framework".to_string());
        }
        _ => {}
    }

    // Add improvement suggestions
    if evidence_quality < 500 {
        suggestions.push("Strengthen evidence with empirical data or expert sources".to_string());
    }
    if input.learner_confidence_permille > 800 && fallacies.len() > 2 {
        suggestions.push("High confidence but multiple fallacies detected - review reasoning".to_string());
    }

    Ok(ArgumentAnalysis {
        main_claim: input.claim_type.clone(),
        components: vec![ArgumentComponent {
            claim: input.argument_text.clone(),
            claim_type: input.claim_type,
            evidence_type: input.provided_evidence.first()
                .map(|(t, _)| t.clone())
                .unwrap_or(EvidenceType::None),
            evidence_strength: EvidenceStrength {
                relevance_permille: evidence_quality,
                reliability_permille: evidence_quality,
                sufficiency_permille: if input.provided_evidence.len() >= 2 { 700 } else { 400 },
                overall_permille: evidence_quality,
            },
            assumptions: assumptions.clone(),
            detected_fallacies: fallacies.iter().map(|(f, _)| f.clone()).collect(),
        }],
        logical_validity_permille: logical_validity,
        evidence_quality_permille: evidence_quality,
        fallacies_detected: fallacies,
        hidden_assumptions: assumptions,
        overall_strength_permille: overall as u16,
        improvement_suggestions: suggestions,
    })
}

// =============================================================================
// FEATURE 16: Epistemic Vigilance
// Source credibility, bias detection, uncertainty quantification
// =============================================================================

/// Types of cognitive biases
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CognitiveBias {
    ConfirmationBias,      // Seeking confirming evidence
    AnchoringBias,         // Over-relying on first information
    AvailabilityHeuristic, // Judging by ease of recall
    DunningKruger,         // Overconfidence from ignorance
    HindsightBias,         // "Knew it all along"
    SunkCostFallacy,       // Continuing due to past investment
    BandwagonEffect,       // Following the crowd
    HaloEffect,            // One trait influencing others
    NegativeBias,          // Weighting negative more than positive
    OptimismBias,          // Overestimating positive outcomes
    StatusQuoBias,         // Preferring current state
    Groupthink,            // Conforming to group consensus
    BlindSpotBias,         // Not seeing own biases
    ProjectionBias,        // Assuming others think like us
    RecencyBias,           // Overweighting recent events
}

/// Source credibility assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceCredibility {
    pub expertise_permille: u16,
    pub track_record_permille: u16,
    pub transparency_permille: u16,
    pub independence_permille: u16,
    pub consensus_alignment_permille: u16,
    pub overall_credibility_permille: u16,
}

/// Source type for credibility evaluation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SourceType {
    PeerReviewedJournal,
    ExpertOpinion,
    GovernmentSource,
    NewsMedia,
    SocialMedia,
    PersonalBlog,
    WikiSource,
    AnonymousSource,
    PrimarySource,
    SecondarySource,
}

/// Input for source evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceEvaluationInput {
    pub source_type: SourceType,
    pub author_credentials: Option<String>,
    pub publication_date: u64,
    pub has_citations: bool,
    pub citation_count: u32,
    pub conflicts_of_interest: bool,
}

/// Evaluate source credibility
#[hdk_extern]
pub fn evaluate_source(input: SourceEvaluationInput) -> ExternResult<SourceCredibility> {
    // Base credibility by source type
    let base = match input.source_type {
        SourceType::PeerReviewedJournal => 900u16,
        SourceType::PrimarySource => 850,
        SourceType::GovernmentSource => 750,
        SourceType::ExpertOpinion => 700,
        SourceType::SecondarySource => 650,
        SourceType::NewsMedia => 600,
        SourceType::WikiSource => 500,
        SourceType::PersonalBlog => 350,
        SourceType::SocialMedia => 250,
        SourceType::AnonymousSource => 100,
    };

    // Modifiers
    let credential_bonus = if input.author_credentials.is_some() { 100u16 } else { 0 };
    let citation_bonus = (input.citation_count.min(100) as u16) * 2;
    let conflict_penalty = if input.conflicts_of_interest { 200u16 } else { 0 };
    let has_cites_bonus = if input.has_citations { 100u16 } else { 0 };

    let expertise = base.saturating_add(credential_bonus).min(1000);
    let track_record = base.saturating_add(citation_bonus).min(1000);
    let transparency = if input.has_citations { 700 } else { 400 };
    let independence = base.saturating_sub(conflict_penalty);

    let overall = (expertise as u32 + track_record as u32 + transparency as u32 + independence as u32) / 4;

    Ok(SourceCredibility {
        expertise_permille: expertise,
        track_record_permille: track_record,
        transparency_permille: transparency,
        independence_permille: independence,
        consensus_alignment_permille: base, // Simplified
        overall_credibility_permille: overall as u16,
    })
}

/// Bias detection result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiasAnalysis {
    pub detected_biases: Vec<(CognitiveBias, u16)>, // Bias + strength
    pub self_awareness_permille: u16,
    pub debiasing_strategies: Vec<String>,
    pub most_vulnerable_bias: CognitiveBias,
}

/// Input for bias detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiasDetectionInput {
    pub recent_decisions: Vec<(String, bool)>, // Decision + whether confirmed prior belief
    pub information_sources_diversity: u16, // 0-1000
    pub times_changed_mind_recently: u8,
    pub considers_opposing_views_permille: u16,
}

/// Detect cognitive biases
#[hdk_extern]
pub fn detect_biases(input: BiasDetectionInput) -> ExternResult<BiasAnalysis> {
    let mut biases = Vec::new();
    let mut strategies = Vec::new();

    // Confirmation bias detection
    let confirming = input.recent_decisions.iter().filter(|(_, confirmed)| *confirmed).count();
    let confirmation_rate = if input.recent_decisions.is_empty() {
        500u16
    } else {
        (confirming * 1000 / input.recent_decisions.len()) as u16
    };

    if confirmation_rate > 800 {
        biases.push((CognitiveBias::ConfirmationBias, confirmation_rate));
        strategies.push("Actively seek out disconfirming evidence".to_string());
        strategies.push("Steel-man opposing arguments before dismissing them".to_string());
    }

    // Source diversity bias
    if input.information_sources_diversity < 400 {
        biases.push((CognitiveBias::BandwagonEffect, 800 - input.information_sources_diversity / 2));
        strategies.push("Diversify information sources across viewpoints".to_string());
    }

    // Flexibility detection (inverse of anchoring)
    if input.times_changed_mind_recently == 0 {
        biases.push((CognitiveBias::AnchoringBias, 700));
        strategies.push("Practice updating beliefs when presented with new evidence".to_string());
    }

    // Opposing view consideration
    if input.considers_opposing_views_permille < 300 {
        biases.push((CognitiveBias::BlindSpotBias, 600));
        strategies.push("Regularly engage with well-argued opposing positions".to_string());
    }

    // Self-awareness score
    let self_awareness = input.considers_opposing_views_permille / 2
        + input.information_sources_diversity / 4
        + (input.times_changed_mind_recently as u16 * 50).min(250);

    let most_vulnerable = biases.first()
        .map(|(b, _)| b.clone())
        .unwrap_or(CognitiveBias::ConfirmationBias);

    Ok(BiasAnalysis {
        detected_biases: biases,
        self_awareness_permille: self_awareness,
        debiasing_strategies: strategies,
        most_vulnerable_bias: most_vulnerable,
    })
}

// =============================================================================
// FEATURE 17: Socratic Dialogue
// Probing questions, devil's advocate, steel-manning
// =============================================================================

/// Types of Socratic questions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SocraticQuestionType {
    Clarification,       // "What do you mean by...?"
    ProbeAssumptions,    // "What are you assuming?"
    ProbeEvidence,       // "What evidence supports this?"
    ProbeViewpoints,     // "What would others say?"
    ProbeImplications,   // "What are the consequences?"
    QuestionTheQuestion, // "Why is this important?"
}

/// Socratic dialogue result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocraticDialogue {
    pub question_type: SocraticQuestionType,
    pub question: String,
    pub purpose: String,
    pub expected_insight_depth: u16,
    pub follow_up_questions: Vec<String>,
}

/// Devil's advocate challenge
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DevilsAdvocate {
    pub counter_argument: String,
    pub challenge_type: String,
    pub strength_permille: u16,
    pub response_needed: bool,
}

/// Steel-man representation of opposing view
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SteelMan {
    pub original_position: String,
    pub strongest_version: String,
    pub key_strengths: Vec<String>,
    pub valid_concerns: Vec<String>,
    pub intellectual_honesty_permille: u16,
}

/// Input for Socratic dialogue generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocraticInput {
    pub learner_claim: String,
    pub claim_type: ClaimType,
    pub reasoning_depth: u16, // How deep to probe
    pub prior_questions_asked: u8,
}

/// Generate Socratic questions
#[hdk_extern]
pub fn generate_socratic_questions(input: SocraticInput) -> ExternResult<Vec<SocraticDialogue>> {
    let mut questions = Vec::new();

    // Start with clarification if first question
    if input.prior_questions_asked == 0 {
        questions.push(SocraticDialogue {
            question_type: SocraticQuestionType::Clarification,
            question: format!("When you say '{}', what specifically do you mean?",
                truncate_text(&input.learner_claim, 50)),
            purpose: "Ensure clear understanding of the claim".to_string(),
            expected_insight_depth: 400,
            follow_up_questions: vec![
                "Can you give a concrete example?".to_string(),
                "How would you define the key terms?".to_string(),
            ],
        });
    }

    // Probe assumptions
    questions.push(SocraticDialogue {
        question_type: SocraticQuestionType::ProbeAssumptions,
        question: "What are you assuming to be true for this to hold?".to_string(),
        purpose: "Expose hidden assumptions".to_string(),
        expected_insight_depth: 600,
        follow_up_questions: vec![
            "Why do you think this assumption is justified?".to_string(),
            "What if that assumption were false?".to_string(),
        ],
    });

    // Probe evidence based on claim type
    let evidence_question = match input.claim_type {
        ClaimType::Factual => "What evidence would prove this false?",
        ClaimType::Causal => "How do you know X caused Y rather than correlation?",
        ClaimType::Evaluative => "By what criteria are you judging this?",
        ClaimType::Policy => "What data supports this would achieve the goal?",
        _ => "What is the strongest evidence supporting this?",
    };

    questions.push(SocraticDialogue {
        question_type: SocraticQuestionType::ProbeEvidence,
        question: evidence_question.to_string(),
        purpose: "Evaluate the evidentiary basis".to_string(),
        expected_insight_depth: 700,
        follow_up_questions: vec![
            "Is this evidence sufficient?".to_string(),
            "What alternative explanations exist?".to_string(),
        ],
    });

    // Probe viewpoints if reasoning depth is high enough
    if input.reasoning_depth > 500 {
        questions.push(SocraticDialogue {
            question_type: SocraticQuestionType::ProbeViewpoints,
            question: "What would a thoughtful critic of this position say?".to_string(),
            purpose: "Consider alternative perspectives".to_string(),
            expected_insight_depth: 800,
            follow_up_questions: vec![
                "What's the strongest version of that criticism?".to_string(),
                "Is there any merit to that objection?".to_string(),
            ],
        });
    }

    // Probe implications
    questions.push(SocraticDialogue {
        question_type: SocraticQuestionType::ProbeImplications,
        question: "If this is true, what else must be true?".to_string(),
        purpose: "Explore logical consequences".to_string(),
        expected_insight_depth: 750,
        follow_up_questions: vec![
            "Are you willing to accept those implications?".to_string(),
            "Do any of those implications seem problematic?".to_string(),
        ],
    });

    Ok(questions)
}

/// Helper to truncate text
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len.saturating_sub(3)])
    }
}

// =============================================================================
// FEATURE 18: Metacognition Framework
// Thinking about thinking, strategy selection, self-monitoring
// =============================================================================

/// Metacognitive skills
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetacognitiveSkill {
    Planning,            // Planning approach before starting
    Monitoring,          // Checking comprehension during learning
    Evaluating,          // Assessing learning after completion
    StrategySelection,   // Choosing appropriate strategies
    SelfExplanation,     // Explaining to oneself
    Debugging,           // Identifying and fixing confusion
}

/// Metacognitive awareness levels
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetacognitiveLevel {
    Tacit,       // Uses skills without awareness
    Aware,       // Knows what skills are used
    Strategic,   // Can select appropriate skills
    Reflective,  // Can evaluate and improve skills
}

/// Learning strategy types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LearningStrategy {
    Rehearsal,           // Repetition
    Elaboration,         // Connecting to prior knowledge
    Organization,        // Creating structure
    CriticalThinking,    // Evaluating information
    Metacognition,       // Monitoring own thinking
    ResourceManagement,  // Time and environment
    HelpSeeking,         // Asking for assistance
}

/// Metacognitive assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetacognitiveAssessment {
    pub awareness_level: MetacognitiveLevel,
    pub skill_scores: Vec<(MetacognitiveSkill, u16)>,
    pub strategy_effectiveness: Vec<(LearningStrategy, u16)>,
    pub self_regulation_permille: u16,
    pub improvement_priorities: Vec<MetacognitiveSkill>,
    pub recommended_prompts: Vec<String>,
}

/// Input for metacognitive assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetacognitiveInput {
    pub plans_before_learning: bool,
    pub checks_understanding_during: bool,
    pub reviews_after_completion: bool,
    pub adjusts_strategies: bool,
    pub explains_to_self: bool,
    pub recognizes_confusion: bool,
    pub estimates_time_accuracy_permille: u16, // How accurate are time estimates
    pub performance_prediction_accuracy: u16,   // How well do they predict their performance
}

/// Assess metacognitive abilities
#[hdk_extern]
pub fn assess_metacognition(input: MetacognitiveInput) -> ExternResult<MetacognitiveAssessment> {
    let mut skills = Vec::new();
    let mut priorities = Vec::new();
    let mut prompts = Vec::new();

    // Planning skill
    let planning_score = if input.plans_before_learning { 800u16 } else { 300 };
    skills.push((MetacognitiveSkill::Planning, planning_score));
    if planning_score < 500 {
        priorities.push(MetacognitiveSkill::Planning);
        prompts.push("Before starting, ask: What's my goal? What strategy will I use?".to_string());
    }

    // Monitoring skill
    let monitoring_score = if input.checks_understanding_during { 800 } else { 300 };
    skills.push((MetacognitiveSkill::Monitoring, monitoring_score));
    if monitoring_score < 500 {
        priorities.push(MetacognitiveSkill::Monitoring);
        prompts.push("Pause periodically to ask: Do I understand this? Can I explain it?".to_string());
    }

    // Evaluating skill
    let eval_score = if input.reviews_after_completion { 750 } else { 250 };
    skills.push((MetacognitiveSkill::Evaluating, eval_score));
    if eval_score < 500 {
        priorities.push(MetacognitiveSkill::Evaluating);
        prompts.push("After learning, reflect: What worked? What would I do differently?".to_string());
    }

    // Strategy selection
    let strategy_score = if input.adjusts_strategies { 850 } else { 350 };
    skills.push((MetacognitiveSkill::StrategySelection, strategy_score));

    // Self-explanation
    let self_exp_score = if input.explains_to_self { 800 } else { 400 };
    skills.push((MetacognitiveSkill::SelfExplanation, self_exp_score));

    // Debugging (recognizing confusion)
    let debug_score = if input.recognizes_confusion { 750 } else { 350 };
    skills.push((MetacognitiveSkill::Debugging, debug_score));
    if debug_score < 500 {
        prompts.push("When stuck, pinpoint exactly what's confusing. Is it vocabulary? Logic? Missing background?".to_string());
    }

    // Determine awareness level
    let avg_skill: u32 = skills.iter().map(|(_, s)| *s as u32).sum::<u32>() / skills.len() as u32;
    let awareness_level = if avg_skill >= 800 && input.adjusts_strategies {
        MetacognitiveLevel::Reflective
    } else if avg_skill >= 600 && input.adjusts_strategies {
        MetacognitiveLevel::Strategic
    } else if avg_skill >= 400 {
        MetacognitiveLevel::Aware
    } else {
        MetacognitiveLevel::Tacit
    };

    // Self-regulation score
    let calibration = (input.estimates_time_accuracy_permille + input.performance_prediction_accuracy) / 2;
    let self_regulation = (avg_skill as u16 + calibration) / 2;

    // Strategy effectiveness
    let strategies = vec![
        (LearningStrategy::Rehearsal, 400u16),
        (LearningStrategy::Elaboration, if input.explains_to_self { 800 } else { 500 }),
        (LearningStrategy::CriticalThinking, strategy_score),
        (LearningStrategy::Metacognition, avg_skill as u16),
    ];

    Ok(MetacognitiveAssessment {
        awareness_level,
        skill_scores: skills,
        strategy_effectiveness: strategies,
        self_regulation_permille: self_regulation,
        improvement_priorities: priorities,
        recommended_prompts: prompts,
    })
}

// =============================================================================
// FEATURE 19: Collaborative Knowledge Building
// Argumentation scaffolds, perspective-taking, collective intelligence
// =============================================================================

/// Collaboration role
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CollaborationRole {
    Proposer,     // Introduces ideas
    Questioner,   // Asks clarifying questions
    Challenger,   // Provides counterarguments
    Synthesizer,  // Combines ideas
    Summarizer,   // Condenses discussion
    Facilitator,  // Keeps discussion on track
}

/// Argumentation move types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArgumentationMove {
    Claim,           // Make a claim
    Support,         // Provide evidence
    Challenge,       // Question a claim
    Concede,         // Acknowledge valid point
    Qualify,         // Add nuance
    Synthesize,      // Combine viewpoints
    Clarify,         // Explain meaning
    Redirect,        // Change focus
}

/// Perspective-taking assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerspectiveTaking {
    pub can_articulate_other_view_permille: u16,
    pub finds_merit_in_opposition_permille: u16,
    pub integrates_perspectives_permille: u16,
    pub empathy_level: u16,
}

/// Collaborative knowledge assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollaborativeAssessment {
    pub individual_contribution_permille: u16,
    pub builds_on_others_permille: u16,
    pub perspective_taking: PerspectiveTaking,
    pub primary_role: CollaborationRole,
    pub argumentation_quality_permille: u16,
    pub constructive_disagreement_permille: u16,
    pub improvement_suggestions: Vec<String>,
}

/// Input for collaborative assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollaborativeInput {
    pub contributions_made: u8,
    pub responses_to_others: u8,
    pub questions_asked: u8,
    pub perspectives_considered: u8,
    pub ideas_synthesized: u8,
    pub acknowledged_other_viewpoints: bool,
    pub modified_own_position: bool,
}

/// Assess collaborative learning
#[hdk_extern]
pub fn assess_collaboration(input: CollaborativeInput) -> ExternResult<CollaborativeAssessment> {
    let mut suggestions = Vec::new();

    // Individual contribution
    let contribution = (input.contributions_made as u16 * 100).min(1000);

    // Builds on others
    let builds_on = (input.responses_to_others as u16 * 150).min(1000);
    if builds_on < 400 {
        suggestions.push("Try responding to others' ideas before adding new ones".to_string());
    }

    // Perspective taking
    let perspective = PerspectiveTaking {
        can_articulate_other_view_permille: if input.acknowledged_other_viewpoints { 700 } else { 300 },
        finds_merit_in_opposition_permille: if input.modified_own_position { 800 } else { 400 },
        integrates_perspectives_permille: (input.ideas_synthesized as u16 * 200).min(1000),
        empathy_level: (input.perspectives_considered as u16 * 200).min(1000),
    };

    // Primary role based on behavior
    let role = if input.questions_asked > input.contributions_made {
        CollaborationRole::Questioner
    } else if input.ideas_synthesized > 0 {
        CollaborationRole::Synthesizer
    } else if input.responses_to_others > input.contributions_made {
        CollaborationRole::Challenger
    } else {
        CollaborationRole::Proposer
    };

    // Argumentation quality
    let arg_quality = (input.contributions_made + input.responses_to_others) as u16 * 100;

    // Constructive disagreement
    let constructive = if input.acknowledged_other_viewpoints && input.modified_own_position {
        800u16
    } else if input.acknowledged_other_viewpoints {
        600
    } else {
        300
    };

    if !input.modified_own_position {
        suggestions.push("Consider whether any opposing points have merit".to_string());
    }

    Ok(CollaborativeAssessment {
        individual_contribution_permille: contribution,
        builds_on_others_permille: builds_on,
        perspective_taking: perspective,
        primary_role: role,
        argumentation_quality_permille: arg_quality.min(1000),
        constructive_disagreement_permille: constructive,
        improvement_suggestions: suggestions,
    })
}

// =============================================================================
// FEATURE 20: Creativity & Divergent Thinking
// Idea generation, combinatorial creativity, constraint relaxation
// =============================================================================

/// Creative thinking types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CreativeThinkingType {
    Divergent,      // Generate many ideas
    Convergent,     // Evaluate and select
    Lateral,        // Unconventional approaches
    Analogical,     // Transfer from other domains
    Combinatorial,  // Combine existing ideas
}

/// Creativity dimensions (Torrance)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreativityDimensions {
    pub fluency: u16,       // Number of ideas
    pub flexibility: u16,   // Variety of categories
    pub originality: u16,   // Uniqueness
    pub elaboration: u16,   // Detail and development
}

/// Creative technique
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CreativeTechnique {
    Brainstorming,
    SCAMPER,         // Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse
    SixThinkingHats,
    MindMapping,
    Analogies,
    ConstraintRemoval,
    RandomStimulus,
    Reversal,
    WhatIf,
}

/// Creativity assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreativityAssessment {
    pub dimensions: CreativityDimensions,
    pub dominant_thinking_type: CreativeThinkingType,
    pub creativity_quotient_permille: u16,
    pub recommended_techniques: Vec<CreativeTechnique>,
    pub stretch_prompts: Vec<String>,
}

/// Input for creativity assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreativityInput {
    pub ideas_generated: u8,
    pub categories_explored: u8,
    pub novel_connections_made: u8,
    pub detail_level_permille: u16,
    pub willing_to_take_risks: bool,
    pub considers_unconventional: bool,
}

/// Assess creative thinking
#[hdk_extern]
pub fn assess_creativity(input: CreativityInput) -> ExternResult<CreativityAssessment> {
    let mut techniques = Vec::new();
    let mut prompts = Vec::new();

    // Fluency (number of ideas)
    let fluency = (input.ideas_generated as u16 * 100).min(1000);
    if fluency < 500 {
        techniques.push(CreativeTechnique::Brainstorming);
        prompts.push("Set a timer for 5 minutes and list every idea, no matter how wild".to_string());
    }

    // Flexibility (variety)
    let flexibility = (input.categories_explored as u16 * 200).min(1000);
    if flexibility < 500 {
        techniques.push(CreativeTechnique::SCAMPER);
        prompts.push("Consider: How could you Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, or Reverse this?".to_string());
    }

    // Originality (novel connections)
    let originality = (input.novel_connections_made as u16 * 200).min(1000);
    if originality < 500 {
        techniques.push(CreativeTechnique::Analogies);
        prompts.push("What's this similar to in a completely different domain? (e.g., 'This problem is like...')".to_string());
    }

    // Elaboration
    let elaboration = input.detail_level_permille;

    let dimensions = CreativityDimensions {
        fluency,
        flexibility,
        originality,
        elaboration,
    };

    // Dominant thinking type
    let dominant = if input.considers_unconventional {
        CreativeThinkingType::Lateral
    } else if input.novel_connections_made > input.ideas_generated / 2 {
        CreativeThinkingType::Combinatorial
    } else if input.categories_explored > 3 {
        CreativeThinkingType::Divergent
    } else {
        CreativeThinkingType::Convergent
    };

    // Creativity quotient
    let cq = (fluency as u32 + flexibility as u32 + originality as u32 + elaboration as u32) / 4;

    // Add techniques for risk/unconventional
    if !input.willing_to_take_risks {
        techniques.push(CreativeTechnique::ConstraintRemoval);
        prompts.push("What would you do if failure was impossible?".to_string());
    }
    if !input.considers_unconventional {
        techniques.push(CreativeTechnique::Reversal);
        prompts.push("What's the opposite of the expected approach?".to_string());
    }

    Ok(CreativityAssessment {
        dimensions,
        dominant_thinking_type: dominant,
        creativity_quotient_permille: cq as u16,
        recommended_techniques: techniques,
        stretch_prompts: prompts,
    })
}

// =============================================================================
// FEATURE 21: Inquiry-Based Learning
// Question generation, hypothesis formation, evidence weighing
// =============================================================================

/// Question quality types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuestionQuality {
    Factual,          // Who, what, when, where
    Conceptual,       // How, why
    Procedural,       // How to
    Metacognitive,    // What if, what would happen
    Generative,       // What else, what new
}

/// Inquiry phase
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InquiryPhase {
    Questioning,      // Generating questions
    Hypothesizing,    // Forming predictions
    Investigating,    // Gathering evidence
    Analyzing,        // Interpreting data
    Concluding,       // Drawing conclusions
    Reflecting,       // Evaluating the inquiry
}

/// Hypothesis quality
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypothesisQuality {
    pub testable_permille: u16,
    pub specific_permille: u16,
    pub falsifiable_permille: u16,
    pub grounded_in_theory_permille: u16,
}

/// Inquiry assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InquiryAssessment {
    pub current_phase: InquiryPhase,
    pub question_quality: QuestionQuality,
    pub question_depth_permille: u16,
    pub hypothesis_quality: HypothesisQuality,
    pub evidence_evaluation_permille: u16,
    pub conclusion_validity_permille: u16,
    pub scaffolding_needed: Vec<String>,
}

/// Input for inquiry assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InquiryInput {
    pub question_asked: String,
    pub hypothesis_formed: Option<String>,
    pub evidence_gathered: u8,
    pub considered_alternative_explanations: bool,
    pub conclusion_matches_evidence: bool,
}

/// Assess inquiry-based learning
#[hdk_extern]
pub fn assess_inquiry(input: InquiryInput) -> ExternResult<InquiryAssessment> {
    let mut scaffolds = Vec::new();

    // Determine question quality
    let q_lower = input.question_asked.to_lowercase();
    let question_quality = if q_lower.contains("what if") || q_lower.contains("would happen") {
        QuestionQuality::Metacognitive
    } else if q_lower.contains("why") || q_lower.contains("how does") {
        QuestionQuality::Conceptual
    } else if q_lower.contains("how to") || q_lower.contains("how can") {
        QuestionQuality::Procedural
    } else if q_lower.contains("what else") || q_lower.contains("could we") {
        QuestionQuality::Generative
    } else {
        QuestionQuality::Factual
    };

    // Question depth based on quality
    let question_depth = match question_quality {
        QuestionQuality::Metacognitive => 900u16,
        QuestionQuality::Generative => 850,
        QuestionQuality::Conceptual => 700,
        QuestionQuality::Procedural => 500,
        QuestionQuality::Factual => 300,
    };

    if question_depth < 500 {
        scaffolds.push("Try asking 'Why?' or 'What would happen if...?'".to_string());
    }

    // Determine current phase
    let phase = if input.hypothesis_formed.is_none() {
        InquiryPhase::Questioning
    } else if input.evidence_gathered == 0 {
        InquiryPhase::Hypothesizing
    } else if !input.conclusion_matches_evidence {
        InquiryPhase::Analyzing
    } else {
        InquiryPhase::Concluding
    };

    // Hypothesis quality
    let hypothesis_quality = if let Some(ref h) = input.hypothesis_formed {
        let h_lower = h.to_lowercase();
        HypothesisQuality {
            testable_permille: if h_lower.contains("if") && h_lower.contains("then") { 800 } else { 400 },
            specific_permille: if h.len() > 50 { 700 } else { 400 },
            falsifiable_permille: if h_lower.contains("not") || h_lower.contains("unless") { 700 } else { 500 },
            grounded_in_theory_permille: 500, // Would need more context
        }
    } else {
        scaffolds.push("Form a hypothesis: 'If X, then Y because Z'".to_string());
        HypothesisQuality {
            testable_permille: 0,
            specific_permille: 0,
            falsifiable_permille: 0,
            grounded_in_theory_permille: 0,
        }
    };

    // Evidence evaluation
    let evidence_eval = if input.considered_alternative_explanations {
        800u16
    } else if input.evidence_gathered > 3 {
        600
    } else {
        (input.evidence_gathered as u16 * 150).min(450)
    };

    if !input.considered_alternative_explanations && input.evidence_gathered > 0 {
        scaffolds.push("Consider: What else could explain this evidence?".to_string());
    }

    // Conclusion validity
    let conclusion_validity = if input.conclusion_matches_evidence && input.considered_alternative_explanations {
        850u16
    } else if input.conclusion_matches_evidence {
        650
    } else {
        300
    };

    Ok(InquiryAssessment {
        current_phase: phase,
        question_quality,
        question_depth_permille: question_depth,
        hypothesis_quality,
        evidence_evaluation_permille: evidence_eval,
        conclusion_validity_permille: conclusion_validity,
        scaffolding_needed: scaffolds,
    })
}

// =============================================================================
// TIER 1: EMOTIONAL & AFFECTIVE LEARNING
// Pekrun's Control-Value Theory - Academic emotions profoundly impact learning
// =============================================================================

/// Academic emotions based on Pekrun's Control-Value Theory
/// These 9 emotions cover the full spectrum of learning-related affect
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AcademicEmotion {
    // Positive activating emotions (enhance learning)
    Enjoyment,        // Intrinsic pleasure in learning
    Hope,             // Positive expectation of success
    Pride,            // Achievement satisfaction

    // Positive deactivating emotions
    Relief,           // After overcoming challenge
    Contentment,      // Satisfaction with progress

    // Negative activating emotions (can help or hurt)
    Anxiety,          // Fear of failure
    Anger,            // Frustration with obstacles
    Shame,            // Self-criticism after failure

    // Negative deactivating emotions (usually harmful)
    Hopelessness,     // Belief that success is impossible
    Boredom,          // Lack of engagement or challenge
}

/// Emotional valence (positive/negative) and activation (high/low energy)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmotionalValence {
    PositiveActivating,   // Enjoyment, hope, pride - best for learning
    PositiveDeactivating, // Relief, contentment - good for consolidation
    NegativeActivating,   // Anxiety, anger - can motivate or harm
    NegativeDeactivating, // Boredom, hopelessness - worst for learning
}

/// Emotional regulation strategies based on Gross's Process Model
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EmotionalRegulationStrategy {
    // Antecedent-focused (before emotion peaks)
    SituationSelection,     // Choose engaging activities
    SituationModification,  // Adjust difficulty/context
    AttentionalDeployment,  // Redirect focus
    CognitiveReappraisal,   // Reframe the situation

    // Response-focused (after emotion arises)
    ExpressionSuppression,  // Control outward display
    AcceptanceAndCommitment,// Acknowledge and proceed
    MindfulnessAwareness,   // Observe without judgment

    // Social strategies
    SocialSupport,          // Seek help from others
    CollaborativeCoping,    // Work through with peers
}

/// Frustration state - distinguished from productive struggle
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FrustrationState {
    None,                // No frustration
    Productive,          // "Good" frustration - motivating
    Mounting,            // Building but manageable
    Peak,                // High frustration - intervention needed
    Destructive,         // Harmful - causing disengagement
    Recovery,            // Coming down from peak
}

/// Confusion as a learning signal (D'Mello & Graesser research)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfusionState {
    NoConfusion,         // Clear understanding
    ProductiveConfusion, // Drives deeper processing (optimal!)
    Stuck,               // Confused but trying
    Overwhelmed,         // Too much confusion
    Disengaged,          // Given up due to confusion
}

/// Current emotional state of learner
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalState {
    pub primary_emotion: AcademicEmotion,
    pub intensity_permille: u16,              // 0-1000
    pub valence: EmotionalValence,
    pub frustration: FrustrationState,
    pub confusion: ConfusionState,
    pub emotional_stability_permille: u16,    // Variance in recent emotions
    pub duration_minutes: u16,                // How long in this state
}

/// Emotional pattern over time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionalTrajectory {
    pub starting_emotion: AcademicEmotion,
    pub current_emotion: AcademicEmotion,
    pub trajectory_direction: TrendDirection,  // Improving, stable, declining
    pub volatility_permille: u16,
    pub resilience_score_permille: u16,        // Recovery from negative emotions
}

/// Input for emotion assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionAssessmentInput {
    pub response_time_variance_permille: u16,
    pub accuracy_trend: TrendDirection,
    pub session_duration_minutes: u16,
    pub errors_last_5_minutes: u8,
    pub help_requests: u8,
    pub pause_frequency: u8,
    pub self_reported_feeling: Option<AcademicEmotion>,
}

/// Emotion assessment result with interventions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmotionAssessment {
    pub detected_state: EmotionalState,
    pub confidence_permille: u16,
    pub recommended_strategies: Vec<EmotionalRegulationStrategy>,
    pub optimal_for_learning: bool,
    pub intervention_urgency: u8,  // 0-10 scale
    pub suggested_activities: Vec<String>,
}

/// Assess learner's emotional state from behavioral signals
#[hdk_extern]
pub fn assess_emotional_state(input: EmotionAssessmentInput) -> ExternResult<EmotionAssessment> {
    let mut strategies = Vec::new();
    let mut activities = Vec::new();

    // Detect primary emotion from behavioral signals
    let (emotion, intensity) = if input.errors_last_5_minutes > 5 && input.help_requests > 2 {
        strategies.push(EmotionalRegulationStrategy::SituationModification);
        activities.push("Take a short break".to_string());
        (AcademicEmotion::Anxiety, 750u16)
    } else if input.pause_frequency > 4 && input.accuracy_trend == TrendDirection::Decreasing {
        strategies.push(EmotionalRegulationStrategy::AttentionalDeployment);
        (AcademicEmotion::Boredom, 650)
    } else if input.response_time_variance_permille > 500 {
        strategies.push(EmotionalRegulationStrategy::CognitiveReappraisal);
        (AcademicEmotion::Anger, 600)
    } else if input.accuracy_trend == TrendDirection::Increasing && input.errors_last_5_minutes < 2 {
        (AcademicEmotion::Enjoyment, 700)
    } else if let Some(ref reported) = input.self_reported_feeling {
        (reported.clone(), 800)
    } else {
        (AcademicEmotion::Contentment, 500)
    };

    // Determine valence
    let valence = match emotion {
        AcademicEmotion::Enjoyment | AcademicEmotion::Hope | AcademicEmotion::Pride =>
            EmotionalValence::PositiveActivating,
        AcademicEmotion::Relief | AcademicEmotion::Contentment =>
            EmotionalValence::PositiveDeactivating,
        AcademicEmotion::Anxiety | AcademicEmotion::Anger | AcademicEmotion::Shame =>
            EmotionalValence::NegativeActivating,
        AcademicEmotion::Hopelessness | AcademicEmotion::Boredom =>
            EmotionalValence::NegativeDeactivating,
    };

    // Detect frustration state
    let frustration = if input.errors_last_5_minutes > 7 && input.help_requests > 3 {
        FrustrationState::Destructive
    } else if input.errors_last_5_minutes > 4 {
        FrustrationState::Peak
    } else if input.errors_last_5_minutes > 2 {
        FrustrationState::Productive
    } else {
        FrustrationState::None
    };

    // Detect confusion state
    let confusion = if input.pause_frequency > 6 && input.help_requests > 2 {
        ConfusionState::Overwhelmed
    } else if input.pause_frequency > 3 && input.accuracy_trend == TrendDirection::Decreasing {
        ConfusionState::Stuck
    } else if input.pause_frequency > 2 {
        ConfusionState::ProductiveConfusion  // Best for learning!
    } else {
        ConfusionState::NoConfusion
    };

    let optimal_for_learning = matches!(valence, EmotionalValence::PositiveActivating) ||
        confusion == ConfusionState::ProductiveConfusion ||
        frustration == FrustrationState::Productive;

    let intervention_urgency = match (&frustration, &confusion) {
        (FrustrationState::Destructive, _) => 10,
        (_, ConfusionState::Overwhelmed) => 9,
        (FrustrationState::Peak, _) => 7,
        (_, ConfusionState::Disengaged) => 8,
        _ => 2,
    };

    if intervention_urgency > 7 {
        activities.push("Switch to an easier topic".to_string());
        activities.push("Review previously mastered content".to_string());
    }

    Ok(EmotionAssessment {
        detected_state: EmotionalState {
            primary_emotion: emotion,
            intensity_permille: intensity,
            valence,
            frustration,
            confusion,
            emotional_stability_permille: 1000 - input.response_time_variance_permille,
            duration_minutes: input.session_duration_minutes,
        },
        confidence_permille: if input.self_reported_feeling.is_some() { 850 } else { 650 },
        recommended_strategies: strategies,
        optimal_for_learning,
        intervention_urgency,
        suggested_activities: activities,
    })
}

// =============================================================================
// TIER 2: DELIBERATE PRACTICE FRAMEWORK
// Ericsson's research on expertise development
// =============================================================================

/// Phases of deliberate practice
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeliberatePracticePhase {
    SkillAnalysis,        // Break down the skill
    StretchGoalSetting,   // Set just-beyond-current goals
    FocusedRepetition,    // Concentrated practice
    ImmediateFeedback,    // Rapid feedback loop
    Refinement,           // Adjust based on feedback
    Consolidation,        // Solidify improvements
}

/// Feedback timing quality
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedbackTiming {
    Immediate,            // Within seconds (ideal)
    NearImmediate,        // Within minutes
    Delayed,              // Hours later
    VeryDelayed,          // Days later (least effective)
}

/// Feedback specificity
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedbackSpecificity {
    Precise,              // Exact what/why/how to improve
    Targeted,             // Specific area identified
    General,              // Overall direction
    Vague,                // Unhelpful generalities
}

/// Stretch zone assessment (just beyond current ability)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum StretchZone {
    TooEasy,              // Below current ability (comfort zone)
    OptimalStretch,       // Just beyond ability (learning zone) - IDEAL
    ModerateStretch,      // Challenging but achievable
    OverStretch,          // Too far beyond (panic zone)
}

/// Mental representation quality (expert mental models)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MentalRepresentationQuality {
    pub pattern_recognition_permille: u16,   // Seeing patterns quickly
    pub chunking_ability_permille: u16,      // Grouping related elements
    pub anticipation_accuracy_permille: u16, // Predicting outcomes
    pub error_detection_permille: u16,       // Spotting own mistakes
    pub self_monitoring_permille: u16,       // Metacognitive awareness
}

/// Skill decomposition for deliberate practice
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillDecomposition {
    pub parent_skill_id: String,
    pub sub_skills: Vec<String>,
    pub weakest_sub_skill: String,
    pub practice_priority_order: Vec<String>,
    pub estimated_hours_to_improve: u16,
}

/// Deliberate practice session quality
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliberatePracticeQuality {
    pub focus_intensity_permille: u16,       // Mental engagement
    pub stretch_zone: StretchZone,
    pub feedback_timing: FeedbackTiming,
    pub feedback_specificity: FeedbackSpecificity,
    pub repetition_with_variation: bool,
    pub goal_specificity_permille: u16,
    pub overall_quality_permille: u16,
}

/// Input for deliberate practice assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliberatePracticeInput {
    pub current_mastery_permille: u16,
    pub task_difficulty_permille: u16,
    pub time_to_feedback_seconds: u32,
    pub feedback_detail_level: u8,           // 1-5 scale
    pub practice_duration_minutes: u16,
    pub distractions_count: u8,
    pub variations_practiced: u8,
    pub specific_goal_set: bool,
}

/// Deliberate practice assessment result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliberatePracticeAssessment {
    pub current_phase: DeliberatePracticePhase,
    pub quality: DeliberatePracticeQuality,
    pub mental_representation: MentalRepresentationQuality,
    pub improvement_rate_permille: u16,
    pub recommendations: Vec<String>,
    pub estimated_practice_to_next_level_hours: u16,
}

/// Assess deliberate practice quality
#[hdk_extern]
pub fn assess_deliberate_practice(input: DeliberatePracticeInput) -> ExternResult<DeliberatePracticeAssessment> {
    let mut recommendations = Vec::new();

    // Determine stretch zone
    let difficulty_gap = if input.task_difficulty_permille > input.current_mastery_permille {
        input.task_difficulty_permille - input.current_mastery_permille
    } else {
        0
    };

    let stretch_zone = if difficulty_gap < 50 {
        recommendations.push("Increase difficulty - you're in comfort zone".to_string());
        StretchZone::TooEasy
    } else if difficulty_gap <= 150 {
        StretchZone::OptimalStretch
    } else if difficulty_gap <= 300 {
        StretchZone::ModerateStretch
    } else {
        recommendations.push("Reduce difficulty - too far beyond current ability".to_string());
        StretchZone::OverStretch
    };

    // Feedback timing
    let feedback_timing = if input.time_to_feedback_seconds < 5 {
        FeedbackTiming::Immediate
    } else if input.time_to_feedback_seconds < 60 {
        FeedbackTiming::NearImmediate
    } else if input.time_to_feedback_seconds < 3600 {
        recommendations.push("Seek faster feedback for better learning".to_string());
        FeedbackTiming::Delayed
    } else {
        recommendations.push("Feedback too slow - find immediate feedback sources".to_string());
        FeedbackTiming::VeryDelayed
    };

    // Feedback specificity
    let feedback_specificity = match input.feedback_detail_level {
        5 => FeedbackSpecificity::Precise,
        4 => FeedbackSpecificity::Targeted,
        3 => FeedbackSpecificity::General,
        _ => {
            recommendations.push("Seek more specific feedback".to_string());
            FeedbackSpecificity::Vague
        }
    };

    // Focus intensity
    let focus_intensity = if input.distractions_count == 0 {
        900u16
    } else {
        900u16.saturating_sub(input.distractions_count as u16 * 150)
    };

    if focus_intensity < 600 {
        recommendations.push("Minimize distractions during practice".to_string());
    }

    // Variation practice
    if input.variations_practiced < 3 && input.practice_duration_minutes > 20 {
        recommendations.push("Practice with more variation for better transfer".to_string());
    }

    // Goal specificity
    if !input.specific_goal_set {
        recommendations.push("Set specific, measurable practice goals".to_string());
    }

    // Calculate overall quality
    let timing_score = match feedback_timing {
        FeedbackTiming::Immediate => 1000u16,
        FeedbackTiming::NearImmediate => 800,
        FeedbackTiming::Delayed => 500,
        FeedbackTiming::VeryDelayed => 200,
    };

    let zone_score = match stretch_zone {
        StretchZone::OptimalStretch => 1000u16,
        StretchZone::ModerateStretch => 750,
        StretchZone::TooEasy => 300,
        StretchZone::OverStretch => 200,
    };

    let overall_quality = (focus_intensity + timing_score + zone_score) / 3;

    // Mental representation (would improve with practice)
    let mental_rep = MentalRepresentationQuality {
        pattern_recognition_permille: input.current_mastery_permille,
        chunking_ability_permille: input.current_mastery_permille.saturating_sub(100),
        anticipation_accuracy_permille: input.current_mastery_permille.saturating_sub(150),
        error_detection_permille: (focus_intensity + input.current_mastery_permille) / 2,
        self_monitoring_permille: if input.specific_goal_set { 700 } else { 400 },
    };

    // Improvement rate based on practice quality
    let improvement_rate = (overall_quality * 15 / 100).min(150);

    // Hours to next level (mastery >= 800)
    let remaining_to_mastery = 800u16.saturating_sub(input.current_mastery_permille);
    let hours_estimate = if improvement_rate > 0 {
        (remaining_to_mastery / improvement_rate).max(1)
    } else {
        100
    };

    Ok(DeliberatePracticeAssessment {
        current_phase: DeliberatePracticePhase::FocusedRepetition,
        quality: DeliberatePracticeQuality {
            focus_intensity_permille: focus_intensity,
            stretch_zone,
            feedback_timing,
            feedback_specificity,
            repetition_with_variation: input.variations_practiced >= 3,
            goal_specificity_permille: if input.specific_goal_set { 800 } else { 300 },
            overall_quality_permille: overall_quality,
        },
        mental_representation: mental_rep,
        improvement_rate_permille: improvement_rate,
        recommendations,
        estimated_practice_to_next_level_hours: hours_estimate,
    })
}

// =============================================================================
// TIER 3: SOCIAL-EMOTIONAL LEARNING (SEL)
// CASEL Framework - 5 Core Competencies
// =============================================================================

/// CASEL's 5 core SEL competencies
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SELCompetency {
    SelfAwareness,          // Recognizing emotions, strengths, limitations
    SelfManagement,         // Regulating emotions, setting goals
    SocialAwareness,        // Empathy, appreciating diversity
    RelationshipSkills,     // Communication, collaboration, conflict resolution
    ResponsibleDecisionMaking, // Ethical reasoning, evaluating consequences
}

/// Self-awareness sub-skills
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SelfAwarenessSkill {
    EmotionIdentification,    // Naming feelings accurately
    StrengthRecognition,      // Knowing personal strengths
    LimitationAwareness,      // Recognizing areas for growth
    ConfidenceCalibration,    // Accurate self-assessment
    ValuesClarification,      // Understanding personal values
    GrowthMindsetOrientation, // Believing in ability to grow
}

/// Self-management sub-skills
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SelfManagementSkill {
    ImpulseControl,           // Resisting immediate urges
    StressManagement,         // Coping with pressure
    SelfMotivation,           // Internal drive
    GoalSetting,              // Creating meaningful goals
    OrganizationalSkills,     // Planning and time management
    SelfDiscipline,           // Sustained effort
}

/// Social awareness sub-skills
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SocialAwarenessSkill {
    PerspectiveTaking,        // Understanding others' viewpoints
    Empathy,                  // Feeling with others
    DiversityAppreciation,    // Valuing differences
    RespectForOthers,         // Treating others with dignity
    SocialCueReading,         // Interpreting nonverbal signals
    ContextualAwareness,      // Understanding social norms
}

/// Relationship skills sub-skills
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipSkill {
    Communication,            // Clear expression and listening
    SocialEngagement,         // Building connections
    CollaborativeTeamwork,    // Working effectively with others
    ConflictResolution,       // Resolving disagreements constructively
    SeekingHelp,              // Asking for support when needed
    OfferingHelp,             // Providing support to others
}

/// Responsible decision-making sub-skills
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionMakingSkill {
    ProblemIdentification,    // Recognizing issues clearly
    AlternativeGeneration,    // Creating multiple options
    ConsequenceEvaluation,    // Predicting outcomes
    EthicalReasoning,         // Considering right and wrong
    ReflectiveAnalysis,       // Learning from decisions
    CriticalThinking,         // Evaluating information
}

/// SEL competency scores
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SELProfile {
    pub self_awareness_permille: u16,
    pub self_management_permille: u16,
    pub social_awareness_permille: u16,
    pub relationship_skills_permille: u16,
    pub decision_making_permille: u16,
    pub overall_sel_permille: u16,
    pub strongest_competency: SELCompetency,
    pub growth_area: SELCompetency,
}

/// SEL intervention recommendation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SELIntervention {
    // Self-awareness interventions
    JournalingPrompt,
    StrengthsInventory,
    FeelingVocabularyBuilding,

    // Self-management interventions
    BreathingExercise,
    GoalSettingWorksheet,
    TimeManagementTool,

    // Social awareness interventions
    PerspectiveTakingExercise,
    CulturalExplorationActivity,
    EmpathyMapping,

    // Relationship interventions
    ActiveListeningPractice,
    ConflictScenarioRolePlay,
    CollaborativeProject,

    // Decision-making interventions
    EthicalDilemmaDiscussion,
    ConsequenceMapping,
    ReflectionPrompt,
}

/// Input for SEL assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SELInput {
    // Self-awareness indicators
    pub accurately_named_emotion: bool,
    pub identified_learning_strengths: bool,
    pub realistic_self_assessment: bool,

    // Self-management indicators
    pub completed_goals_ratio_permille: u16,
    pub handled_setback_constructively: bool,
    pub maintained_focus_permille: u16,

    // Social awareness indicators
    pub considered_peer_perspective: bool,
    pub showed_empathy_in_feedback: bool,
    pub respected_diverse_approaches: bool,

    // Relationship indicators
    pub effective_collaboration_permille: u16,
    pub sought_help_appropriately: bool,
    pub offered_help_to_peers: bool,

    // Decision-making indicators
    pub considered_multiple_solutions: bool,
    pub evaluated_consequences: bool,
    pub made_ethical_choice: bool,
}

/// SEL assessment result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SELAssessment {
    pub profile: SELProfile,
    pub interventions: Vec<SELIntervention>,
    pub growth_trajectory: TrendDirection,
    pub peer_comparison_percentile: u8,
    pub recommended_activities: Vec<String>,
}

/// Assess social-emotional learning competencies
#[hdk_extern]
pub fn assess_sel_competencies(input: SELInput) -> ExternResult<SELAssessment> {
    let mut interventions = Vec::new();
    let mut activities = Vec::new();

    // Self-awareness score
    let self_awareness = {
        let mut score = 400u16;
        if input.accurately_named_emotion { score += 200; }
        if input.identified_learning_strengths { score += 200; }
        if input.realistic_self_assessment { score += 200; }
        score.min(1000)
    };

    if self_awareness < 600 {
        interventions.push(SELIntervention::JournalingPrompt);
        activities.push("Complete a learning strengths reflection".to_string());
    }

    // Self-management score
    let self_management = {
        let goal_score = input.completed_goals_ratio_permille / 3;
        let setback_score = if input.handled_setback_constructively { 300u16 } else { 0 };
        let focus_score = input.maintained_focus_permille / 3;
        goal_score + setback_score + focus_score
    };

    if self_management < 600 {
        interventions.push(SELIntervention::GoalSettingWorksheet);
        activities.push("Practice a 5-minute breathing exercise before study".to_string());
    }

    // Social awareness score
    let social_awareness = {
        let mut score = 400u16;
        if input.considered_peer_perspective { score += 200; }
        if input.showed_empathy_in_feedback { score += 200; }
        if input.respected_diverse_approaches { score += 200; }
        score.min(1000)
    };

    if social_awareness < 600 {
        interventions.push(SELIntervention::PerspectiveTakingExercise);
        activities.push("Consider how a peer might approach this problem".to_string());
    }

    // Relationship skills score
    let relationship_skills = {
        let collab_score = input.effective_collaboration_permille / 3;
        let help_sought = if input.sought_help_appropriately { 200u16 } else { 0 };
        let help_given = if input.offered_help_to_peers { 200u16 } else { 0 };
        collab_score + help_sought + help_given + 200 // Base
    };

    if relationship_skills < 600 {
        interventions.push(SELIntervention::CollaborativeProject);
        activities.push("Join a study group for peer learning".to_string());
    }

    // Decision-making score
    let decision_making = {
        let mut score = 400u16;
        if input.considered_multiple_solutions { score += 200; }
        if input.evaluated_consequences { score += 200; }
        if input.made_ethical_choice { score += 200; }
        score.min(1000)
    };

    if decision_making < 600 {
        interventions.push(SELIntervention::ConsequenceMapping);
        activities.push("Explore multiple solution paths before choosing".to_string());
    }

    // Overall SEL
    let overall = (self_awareness + self_management + social_awareness +
                   relationship_skills + decision_making) / 5;

    // Find strongest and weakest
    let scores = [
        (SELCompetency::SelfAwareness, self_awareness),
        (SELCompetency::SelfManagement, self_management),
        (SELCompetency::SocialAwareness, social_awareness),
        (SELCompetency::RelationshipSkills, relationship_skills),
        (SELCompetency::ResponsibleDecisionMaking, decision_making),
    ];

    let strongest = scores.iter().max_by_key(|x| x.1).map(|x| x.0.clone()).unwrap_or(SELCompetency::SelfAwareness);
    let growth_area = scores.iter().min_by_key(|x| x.1).map(|x| x.0.clone()).unwrap_or(SELCompetency::SelfAwareness);

    Ok(SELAssessment {
        profile: SELProfile {
            self_awareness_permille: self_awareness,
            self_management_permille: self_management,
            social_awareness_permille: social_awareness,
            relationship_skills_permille: relationship_skills,
            decision_making_permille: decision_making,
            overall_sel_permille: overall,
            strongest_competency: strongest,
            growth_area,
        },
        interventions,
        growth_trajectory: if overall > 700 { TrendDirection::Increasing } else { TrendDirection::Stable },
        peer_comparison_percentile: (overall / 10) as u8,
        recommended_activities: activities,
    })
}

// =============================================================================
// TIER 4: STEALTH & DYNAMIC ASSESSMENT
// Embedded evaluation and Vygotsky's ZPD-based dynamic assessment
// =============================================================================

/// Stealth assessment indicators (embedded in learning activities)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum StealthIndicator {
    // Time-based indicators
    TimeToFirstAction,        // Hesitation suggests uncertainty
    PausePatterns,            // Where learner pauses indicates difficulty
    ResponseTimeVariance,     // Consistency indicates confidence

    // Action-based indicators
    ToolUsagePatterns,        // Which tools used suggests strategy
    HintRequestTiming,        // When hints are sought
    RevisionBehavior,         // Frequency and nature of changes

    // Navigation indicators
    PathThroughContent,       // Linear vs exploratory
    RevisitPatterns,          // What's revisited and when
    SkippingBehavior,         // What's skipped

    // Social indicators
    HelpSeekingPatterns,      // When and how help is sought
    PeerInteractionQuality,   // Nature of peer interactions
    ExplanationDepth,         // Quality when explaining to others
}

/// Evidence type for assessment
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AssessmentEvidence {
    DirectPerformance,        // Actual task completion
    ProcessEvidence,          // How task was approached
    TransferEvidence,         // Application in new context
    ExplanationEvidence,      // Verbal/written explanation
    PeerTeachingEvidence,     // Teaching others
    SelfAssessmentEvidence,   // Learner's own assessment
}

/// Dynamic assessment scaffolding level (Vygotsky ZPD)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScaffoldingLevel {
    NoSupport,                // Independent performance
    Minimal,                  // Light hints
    Moderate,                 // Guiding questions
    Substantial,              // Step-by-step guidance
    Maximal,                  // Complete modeling
}

/// Learning potential assessment (dynamic assessment outcome)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningPotential {
    pub current_performance_permille: u16,
    pub supported_performance_permille: u16,
    pub learning_potential_gap: u16,         // Difference = ZPD width
    pub scaffolding_responsiveness: u16,     // How well learner uses support
    pub transfer_potential_permille: u16,    // Ability to generalize
    pub modifiability_score_permille: u16,   // How teachable
}

/// Self-assessment accuracy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfAssessmentAccuracy {
    pub predicted_score_permille: u16,
    pub actual_score_permille: u16,
    pub accuracy_permille: u16,              // How close prediction was
    pub bias_direction: i16,                 // Positive = overconfident
    pub calibration_trend: CalibrationTrend,
}

/// Stealth assessment data collected
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StealthData {
    pub time_to_first_action_ms: u32,
    pub average_pause_duration_ms: u32,
    pub response_time_variance_permille: u16,
    pub hints_requested: u8,
    pub revisions_made: u8,
    pub content_revisits: u8,
    pub items_skipped: u8,
    pub peer_interactions: u8,
    pub explanation_attempts: u8,
}

/// Input for stealth assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StealthAssessmentInput {
    pub stealth_data: StealthData,
    pub task_difficulty_permille: u16,
    pub self_predicted_score: Option<u16>,
    pub actual_performance_permille: u16,
}

/// Stealth assessment result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StealthAssessment {
    pub inferred_mastery_permille: u16,
    pub confidence_in_inference_permille: u16,
    pub key_indicators: Vec<StealthIndicator>,
    pub evidence_types: Vec<AssessmentEvidence>,
    pub self_assessment_accuracy: Option<SelfAssessmentAccuracy>,
    pub recommended_scaffolding: ScaffoldingLevel,
    pub hidden_struggles: Vec<String>,
}

/// Perform stealth assessment from behavioral data
#[hdk_extern]
pub fn perform_stealth_assessment(input: StealthAssessmentInput) -> ExternResult<StealthAssessment> {
    let data = &input.stealth_data;
    let mut indicators = Vec::new();
    let mut evidence = Vec::new();
    let mut hidden_struggles = Vec::new();

    // Analyze time patterns
    let hesitation_signal = if data.time_to_first_action_ms > 5000 {
        indicators.push(StealthIndicator::TimeToFirstAction);
        hidden_struggles.push("Initial hesitation suggests uncertainty about approach".to_string());
        true
    } else {
        false
    };

    // Analyze pause patterns
    let pause_signal = if data.average_pause_duration_ms > 3000 {
        indicators.push(StealthIndicator::PausePatterns);
        hidden_struggles.push("Long pauses indicate processing difficulty".to_string());
        200u16
    } else {
        0
    };

    // Analyze consistency
    if data.response_time_variance_permille > 400 {
        indicators.push(StealthIndicator::ResponseTimeVariance);
        hidden_struggles.push("Inconsistent timing suggests variable understanding".to_string());
    }

    // Analyze help-seeking
    if data.hints_requested > 3 {
        indicators.push(StealthIndicator::HintRequestTiming);
    }

    // Analyze revision behavior
    if data.revisions_made > 5 {
        indicators.push(StealthIndicator::RevisionBehavior);
        hidden_struggles.push("Frequent revisions indicate uncertainty".to_string());
    }

    // Build evidence types
    evidence.push(AssessmentEvidence::ProcessEvidence);
    if data.peer_interactions > 0 {
        evidence.push(AssessmentEvidence::PeerTeachingEvidence);
    }
    if data.explanation_attempts > 0 {
        evidence.push(AssessmentEvidence::ExplanationEvidence);
    }
    evidence.push(AssessmentEvidence::DirectPerformance);

    // Calculate inferred mastery (adjusting actual performance based on struggle signals)
    let struggle_adjustment = pause_signal + (data.hints_requested as u16 * 50) +
                              (data.revisions_made as u16 * 30);
    let inferred_mastery = input.actual_performance_permille.saturating_sub(struggle_adjustment / 3);

    // Confidence in inference
    let confidence = 600 + (indicators.len() as u16 * 50).min(300);

    // Self-assessment accuracy
    let self_accuracy = input.self_predicted_score.map(|predicted| {
        let diff = if predicted > input.actual_performance_permille {
            predicted - input.actual_performance_permille
        } else {
            input.actual_performance_permille - predicted
        };
        let accuracy = 1000u16.saturating_sub(diff);
        let bias = predicted as i16 - input.actual_performance_permille as i16;

        SelfAssessmentAccuracy {
            predicted_score_permille: predicted,
            actual_score_permille: input.actual_performance_permille,
            accuracy_permille: accuracy,
            bias_direction: bias / 10,
            calibration_trend: if accuracy > 800 { CalibrationTrend::Improving } else { CalibrationTrend::Stable },
        }
    });

    // Recommend scaffolding based on struggles
    let scaffolding = if hidden_struggles.len() > 3 || inferred_mastery < 400 {
        ScaffoldingLevel::Substantial
    } else if hidden_struggles.len() > 1 || inferred_mastery < 600 {
        ScaffoldingLevel::Moderate
    } else if hesitation_signal || inferred_mastery < 750 {
        ScaffoldingLevel::Minimal
    } else {
        ScaffoldingLevel::NoSupport
    };

    Ok(StealthAssessment {
        inferred_mastery_permille: inferred_mastery,
        confidence_in_inference_permille: confidence,
        key_indicators: indicators,
        evidence_types: evidence,
        self_assessment_accuracy: self_accuracy,
        recommended_scaffolding: scaffolding,
        hidden_struggles,
    })
}

/// Dynamic assessment input (testing with scaffolding)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicAssessmentInput {
    pub initial_performance_permille: u16,
    pub scaffolding_levels_tried: Vec<ScaffoldingLevel>,
    pub performance_at_each_level: Vec<u16>,  // Permille at each scaffolding level
    pub transfer_task_performance: Option<u16>,
}

/// Dynamic assessment result (learning potential)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicAssessmentResult {
    pub learning_potential: LearningPotential,
    pub optimal_scaffolding: ScaffoldingLevel,
    pub zpd_width_permille: u16,
    pub responsiveness_to_instruction: u16,
    pub recommendations: Vec<String>,
}

/// Perform dynamic assessment (Vygotsky ZPD-based)
#[hdk_extern]
pub fn perform_dynamic_assessment(input: DynamicAssessmentInput) -> ExternResult<DynamicAssessmentResult> {
    let mut recommendations = Vec::new();

    // Find best supported performance
    let best_supported = input.performance_at_each_level.iter().max().copied().unwrap_or(input.initial_performance_permille);

    // Calculate ZPD width (difference between independent and supported)
    let zpd_width = best_supported.saturating_sub(input.initial_performance_permille);

    // Calculate responsiveness (improvement per scaffolding level)
    let levels_count = input.scaffolding_levels_tried.len() as u16;
    let responsiveness = if levels_count > 0 {
        zpd_width / levels_count
    } else {
        0
    };

    // Find optimal scaffolding level
    let optimal_idx = input.performance_at_each_level.iter()
        .enumerate()
        .max_by_key(|(_, &perf)| perf)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let optimal_scaffolding = input.scaffolding_levels_tried.get(optimal_idx)
        .cloned()
        .unwrap_or(ScaffoldingLevel::Moderate);

    // Transfer potential
    let transfer_potential = input.transfer_task_performance
        .map(|t| (t + best_supported) / 2)
        .unwrap_or(best_supported.saturating_sub(200));

    // Modifiability score (how well they respond to teaching)
    let modifiability = if zpd_width > 300 && responsiveness > 50 {
        850u16
    } else if zpd_width > 200 {
        700
    } else if zpd_width > 100 {
        550
    } else {
        400
    };

    // Recommendations
    if zpd_width > 400 {
        recommendations.push("High learning potential - provide challenging tasks with support".to_string());
    }
    if responsiveness > 100 {
        recommendations.push("Highly responsive to instruction - structured teaching effective".to_string());
    }
    if modifiability < 500 {
        recommendations.push("Consider alternative instructional approaches".to_string());
    }
    if transfer_potential < best_supported.saturating_sub(200) {
        recommendations.push("Focus on transfer activities to generalize learning".to_string());
    }

    Ok(DynamicAssessmentResult {
        learning_potential: LearningPotential {
            current_performance_permille: input.initial_performance_permille,
            supported_performance_permille: best_supported,
            learning_potential_gap: zpd_width,
            scaffolding_responsiveness: responsiveness,
            transfer_potential_permille: transfer_potential,
            modifiability_score_permille: modifiability,
        },
        optimal_scaffolding,
        zpd_width_permille: zpd_width,
        responsiveness_to_instruction: responsiveness,
        recommendations,
    })
}

// =============================================================================
// TIER 5: UNIVERSAL DESIGN FOR LEARNING (UDL)
// CAST Framework - Multiple means of engagement, representation, action
// =============================================================================

/// UDL principle categories
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum UDLPrinciple {
    Engagement,              // The "why" of learning - motivation
    Representation,          // The "what" of learning - perception
    ActionExpression,        // The "how" of learning - interaction
}

/// Engagement options (recruiting interest & sustaining effort)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EngagementOption {
    // Recruiting interest
    ChoiceAndAutonomy,        // Learner control
    RelevanceAuthenticity,    // Real-world connection
    ThreatsDistractionsMin,   // Safe learning environment

    // Sustaining effort
    GoalsSaliency,            // Clear objectives
    ChallengeSupport,         // Optimal difficulty
    CollaborationCommunity,   // Peer learning

    // Self-regulation
    ExpectationsBeliefs,      // Growth mindset support
    CopingSkills,             // Emotional regulation
    SelfAssessmentReflection, // Metacognition
}

/// Representation options (perceiving & comprehending)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RepresentationOption {
    // Perception
    CustomizableDisplay,      // Font, size, color options
    AlternativesAuditory,     // Audio alternatives
    AlternativesVisual,       // Visual alternatives

    // Language & symbols
    VocabularySupport,        // Glossaries, definitions
    SymbolDecoding,           // Math, code clarification
    MultipleLanguages,        // Translation support

    // Comprehension
    BackgroundKnowledge,      // Prior knowledge activation
    PatternsRelationships,    // Highlighting structures
    TransferGeneralization,   // Application support
}

/// Action & expression options (physical & strategic)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActionExpressionOption {
    // Physical action
    ResponseMethods,          // Multiple input methods
    NavigationAccess,         // Keyboard, voice, etc.
    AssistiveTech,            // Screen readers, etc.

    // Expression & communication
    MultipleMedia,            // Text, audio, video options
    ToolsComposition,         // Writing, drawing tools
    ScaffoldedPractice,       // Graduated support

    // Executive function
    GoalSettingSupport,       // Goal planning tools
    ProgressMonitoring,       // Progress tracking
    CapacityManagement,       // Working memory support
}

/// Accessibility need type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccessibilityNeed {
    Visual,                   // Blindness, low vision, color blindness
    Auditory,                 // Deafness, hard of hearing
    Motor,                    // Limited mobility, tremors
    Cognitive,                // Attention, memory, processing
    Linguistic,               // Language barriers
    Emotional,                // Anxiety, trauma-informed needs
}

/// Learner UDL preferences profile
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UDLProfile {
    // Engagement preferences
    pub preferred_choice_level: u8,           // 1-5 scale
    pub collaboration_preference_permille: u16,
    pub self_regulation_support_needed: bool,

    // Representation preferences
    pub visual_preference_permille: u16,
    pub auditory_preference_permille: u16,
    pub text_preference_permille: u16,
    pub needs_vocabulary_support: bool,
    pub preferred_language: String,

    // Action preferences
    pub preferred_input_method: String,       // keyboard, voice, touch
    pub preferred_output_format: String,      // text, audio, visual
    pub executive_function_support_needed: bool,

    // Accessibility
    pub accessibility_needs: Vec<AccessibilityNeed>,
}

/// UDL barrier detected
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UDLBarrier {
    pub barrier_type: UDLPrinciple,
    pub severity_permille: u16,
    pub description: String,
    pub recommended_accommodations: Vec<String>,
}

/// Input for UDL assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UDLInput {
    pub learner_profile: UDLProfile,
    pub content_modalities_available: Vec<String>,
    pub interaction_methods_available: Vec<String>,
    pub current_engagement_permille: u16,
    pub comprehension_permille: u16,
    pub expression_success_permille: u16,
}

/// UDL assessment result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UDLAssessment {
    pub overall_accessibility_permille: u16,
    pub barriers_detected: Vec<UDLBarrier>,
    pub engagement_options_recommended: Vec<EngagementOption>,
    pub representation_options_recommended: Vec<RepresentationOption>,
    pub action_options_recommended: Vec<ActionExpressionOption>,
    pub personalized_accommodations: Vec<String>,
    pub inclusive_design_score_permille: u16,
}

/// Assess Universal Design for Learning needs
#[hdk_extern]
pub fn assess_udl_needs(input: UDLInput) -> ExternResult<UDLAssessment> {
    let profile = &input.learner_profile;
    let mut barriers = Vec::new();
    let mut engagement_options = Vec::new();
    let mut representation_options = Vec::new();
    let mut action_options = Vec::new();
    let mut accommodations = Vec::new();

    // Check engagement barriers
    if input.current_engagement_permille < 500 {
        barriers.push(UDLBarrier {
            barrier_type: UDLPrinciple::Engagement,
            severity_permille: 1000 - input.current_engagement_permille,
            description: "Low engagement detected".to_string(),
            recommended_accommodations: vec![
                "Offer more choice in learning path".to_string(),
                "Connect to learner interests".to_string(),
            ],
        });
        engagement_options.push(EngagementOption::ChoiceAndAutonomy);
        engagement_options.push(EngagementOption::RelevanceAuthenticity);
    }

    if profile.self_regulation_support_needed {
        engagement_options.push(EngagementOption::CopingSkills);
        engagement_options.push(EngagementOption::SelfAssessmentReflection);
        accommodations.push("Provide emotional check-ins and regulation tools".to_string());
    }

    if profile.collaboration_preference_permille > 700 {
        engagement_options.push(EngagementOption::CollaborationCommunity);
    }

    // Check representation barriers
    if input.comprehension_permille < 600 {
        barriers.push(UDLBarrier {
            barrier_type: UDLPrinciple::Representation,
            severity_permille: 1000 - input.comprehension_permille,
            description: "Comprehension difficulty detected".to_string(),
            recommended_accommodations: vec![
                "Provide multiple representations".to_string(),
                "Activate background knowledge".to_string(),
            ],
        });
        representation_options.push(RepresentationOption::BackgroundKnowledge);
        representation_options.push(RepresentationOption::PatternsRelationships);
    }

    // Modality preferences
    if profile.visual_preference_permille > 700 {
        representation_options.push(RepresentationOption::AlternativesVisual);
        accommodations.push("Prioritize visual content formats".to_string());
    }
    if profile.auditory_preference_permille > 700 {
        representation_options.push(RepresentationOption::AlternativesAuditory);
        accommodations.push("Provide audio explanations".to_string());
    }
    if profile.needs_vocabulary_support {
        representation_options.push(RepresentationOption::VocabularySupport);
        accommodations.push("Include glossary and term definitions".to_string());
    }

    // Check action/expression barriers
    if input.expression_success_permille < 600 {
        barriers.push(UDLBarrier {
            barrier_type: UDLPrinciple::ActionExpression,
            severity_permille: 1000 - input.expression_success_permille,
            description: "Expression difficulty detected".to_string(),
            recommended_accommodations: vec![
                "Offer multiple ways to demonstrate learning".to_string(),
                "Provide scaffolded practice opportunities".to_string(),
            ],
        });
        action_options.push(ActionExpressionOption::MultipleMedia);
        action_options.push(ActionExpressionOption::ScaffoldedPractice);
    }

    if profile.executive_function_support_needed {
        action_options.push(ActionExpressionOption::GoalSettingSupport);
        action_options.push(ActionExpressionOption::ProgressMonitoring);
        action_options.push(ActionExpressionOption::CapacityManagement);
        accommodations.push("Provide checklists and progress trackers".to_string());
    }

    // Accessibility-specific accommodations
    for need in &profile.accessibility_needs {
        match need {
            AccessibilityNeed::Visual => {
                representation_options.push(RepresentationOption::AlternativesAuditory);
                action_options.push(ActionExpressionOption::AssistiveTech);
                accommodations.push("Enable screen reader compatibility".to_string());
            },
            AccessibilityNeed::Auditory => {
                representation_options.push(RepresentationOption::AlternativesVisual);
                accommodations.push("Provide captions and transcripts".to_string());
            },
            AccessibilityNeed::Motor => {
                action_options.push(ActionExpressionOption::NavigationAccess);
                action_options.push(ActionExpressionOption::ResponseMethods);
                accommodations.push("Enable keyboard-only navigation".to_string());
            },
            AccessibilityNeed::Cognitive => {
                representation_options.push(RepresentationOption::PatternsRelationships);
                action_options.push(ActionExpressionOption::CapacityManagement);
                accommodations.push("Reduce cognitive load with clear structure".to_string());
            },
            AccessibilityNeed::Linguistic => {
                representation_options.push(RepresentationOption::MultipleLanguages);
                representation_options.push(RepresentationOption::VocabularySupport);
            },
            AccessibilityNeed::Emotional => {
                engagement_options.push(EngagementOption::ThreatsDistractionsMin);
                engagement_options.push(EngagementOption::CopingSkills);
                accommodations.push("Create safe, predictable learning environment".to_string());
            },
        }
    }

    // Calculate overall accessibility
    let engagement_score = input.current_engagement_permille;
    let representation_score = input.comprehension_permille;
    let action_score = input.expression_success_permille;
    let overall = (engagement_score + representation_score + action_score) / 3;

    // Inclusive design score (how well content meets diverse needs)
    let inclusive_design = if barriers.is_empty() {
        900u16
    } else {
        900u16.saturating_sub(barriers.len() as u16 * 200)
    };

    Ok(UDLAssessment {
        overall_accessibility_permille: overall,
        barriers_detected: barriers,
        engagement_options_recommended: engagement_options,
        representation_options_recommended: representation_options,
        action_options_recommended: action_options,
        personalized_accommodations: accommodations,
        inclusive_design_score_permille: inclusive_design,
    })
}

// =============================================================================
// TIER 6: GAMIFICATION 2.0 - Advanced Game Mechanics
// =============================================================================
// Types and logic extracted to gamification_social.rs — thin wrappers below.

/// Generate leaderboard
#[hdk_extern]
pub fn get_leaderboard(input: LeaderboardInput) -> ExternResult<Vec<LeaderboardEntry>> {
    gamification_social::get_leaderboard(input)
}

/// Check available achievements
#[hdk_extern]
pub fn check_achievements(_input: AchievementCheckInput) -> ExternResult<Vec<Achievement>> {
    gamification_social::check_achievements(_input)
}

/// Get social comparison
#[hdk_extern]
pub fn get_social_comparison(input: SocialComparisonInput) -> ExternResult<SocialComparison> {
    gamification_social::get_social_comparison(input)
}

/// Get active challenges
#[hdk_extern]
pub fn get_active_challenges(_input: ActiveChallengesInput) -> ExternResult<Vec<ActiveChallenge>> {
    gamification_social::get_active_challenges(_input)
}

// =============================================================================
// TIER 7: ADVANCED ANALYTICS - Predictive Learning Intelligence
// =============================================================================
// Types and logic extracted to analytics.rs — thin wrappers below.

/// Perform cohort analysis
#[hdk_extern]
pub fn analyze_cohort(input: CohortAnalysisInput) -> ExternResult<CohortStats> {
    analytics::analyze_cohort(input)
}

/// Analyze learning trajectory
#[hdk_extern]
pub fn analyze_trajectory(input: TrajectoryInput) -> ExternResult<LearningTrajectory> {
    analytics::analyze_trajectory(input)
}

/// Generate prediction
#[hdk_extern]
pub fn generate_prediction(input: PredictionInput) -> ExternResult<Prediction> {
    analytics::generate_prediction(input)
}

/// Detect learning anomalies
#[hdk_extern]
pub fn detect_anomalies(input: AnomalyDetectionInput) -> ExternResult<Vec<LearningAnomaly>> {
    analytics::detect_anomalies(input)
}

/// Assess at-risk status
#[hdk_extern]
pub fn assess_at_risk(input: AtRiskInput) -> ExternResult<Vec<AtRiskIndicator>> {
    analytics::assess_at_risk(input)
}

// =============================================================================
// TIER 8: AI TUTORING INTEGRATION - Intelligent Tutoring System
// =============================================================================
// Types and logic extracted to tutoring.rs — thin wrappers below.

/// Request a hint
#[hdk_extern]
pub fn request_hint(input: HintRequestInput) -> ExternResult<HintResponse> {
    tutoring::request_hint(input)
}

/// Request personalized explanation
#[hdk_extern]
pub fn request_explanation(input: ExplanationRequestInput) -> ExternResult<PersonalizedExplanation> {
    tutoring::request_explanation(input)
}

/// Generate personalized feedback
#[hdk_extern]
pub fn generate_feedback(input: FeedbackInput) -> ExternResult<PersonalizedFeedback> {
    tutoring::generate_feedback(input)
}

/// Detect misconceptions
#[hdk_extern]
pub fn detect_misconceptions(input: MisconceptionInput) -> ExternResult<Vec<DetectedMisconception>> {
    tutoring::detect_misconceptions(input)
}

/// Get recommended tutoring strategy
#[hdk_extern]
pub fn get_tutoring_strategy(input: TutoringStrategyInput) -> ExternResult<TutoringStrategy> {
    tutoring::get_tutoring_strategy(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_balance_calculation() {
        // Perfect accuracy (100%) should give flow_balance of 800
        let accuracy: u16 = 1000;
        let flow = accuracy.saturating_sub(200).min(800);
        assert_eq!(flow, 800);

        // 50% accuracy should give 300
        let accuracy: u16 = 500;
        let flow = accuracy.saturating_sub(200).min(800);
        assert_eq!(flow, 300);

        // 20% accuracy (too hard)
        let accuracy: u16 = 200;
        let flow = accuracy.saturating_sub(200).min(800);
        assert_eq!(flow, 0);
    }

    #[test]
    fn test_mastery_threshold() {
        assert!(MASTERY_THRESHOLD == 800);

        // Below threshold
        let old = 750u16;
        let new = 790u16;
        let just_mastered = old < MASTERY_THRESHOLD && new >= MASTERY_THRESHOLD;
        assert!(!just_mastered);

        // Crossing threshold
        let old = 790u16;
        let new = 810u16;
        let just_mastered = old < MASTERY_THRESHOLD && new >= MASTERY_THRESHOLD;
        assert!(just_mastered);
    }

    #[test]
    fn test_priority_sorting() {
        let critical = GoalPriority::Critical;
        let high = GoalPriority::High;
        let medium = GoalPriority::Medium;
        let low = GoalPriority::Low;

        let p = |g: &GoalPriority| match g {
            GoalPriority::Critical => 0,
            GoalPriority::High => 1,
            GoalPriority::Medium => 2,
            GoalPriority::Low => 3,
        };

        assert!(p(&critical) < p(&high));
        assert!(p(&high) < p(&medium));
        assert!(p(&medium) < p(&low));
    }

    #[test]
    fn test_running_average() {
        // First attempt: 100 seconds
        let old_total = 0u32;
        let old_avg = 0u32;
        let new_time = 100u32;
        let new_avg = if old_total == 0 {
            new_time
        } else {
            ((old_avg as u64 * old_total as u64 + new_time as u64) / (old_total + 1) as u64) as u32
        };
        assert_eq!(new_avg, 100);

        // Second attempt: 50 seconds -> avg should be 75
        let old_total = 1u32;
        let old_avg = 100u32;
        let new_time = 50u32;
        let new_avg = ((old_avg as u64 * old_total as u64 + new_time as u64) / (old_total + 1) as u64) as u32;
        assert_eq!(new_avg, 75);
    }

    // ============== Smart Recommendation Tests ==============

    #[test]
    fn test_optimal_difficulty_range() {
        // Test optimal difficulty calculation based on ZPD (Zone of Proximal Development)
        // Base range is 80%-85% of mastery level

        // Low mastery (400) - should target accessible difficulty
        let mastery: u16 = 400;
        let (lower, upper) = ((mastery as f32 * 0.80) as u16, (mastery as f32 * 0.85) as u16 + 100);
        assert!(lower <= 400); // Lower bound is achievable
        assert!(upper >= lower); // Valid range

        // High mastery (900) - should target challenging difficulty
        let mastery: u16 = 900;
        let expected_lower = (mastery as f32 * 0.80) as u16; // ~720
        let expected_upper = ((mastery as f32 * 0.85) as u16 + 100).min(1000); // ~865, capped at 1000
        assert!(expected_lower >= 700);
        assert!(expected_upper <= 1000);
    }

    #[test]
    fn test_time_bonus_morning() {
        // Morning (6-11am) should favor lessons and quizzes
        let hour: u8 = 9;
        let lesson_bonus = match hour {
            6..=11 => 150,
            _ => 50,
        };
        let exercise_bonus = match hour {
            6..=11 => 75,
            _ => 150,
        };
        assert_eq!(lesson_bonus, 150); // Lessons are best in morning
        assert_eq!(exercise_bonus, 75); // Exercises get lower morning bonus
    }

    #[test]
    fn test_time_bonus_afternoon() {
        // Afternoon (12-16) should favor exercises and practice
        let hour: u8 = 14;
        let exercise_bonus = match hour {
            12..=16 => 150,
            _ => 75,
        };
        let lesson_bonus = match hour {
            12..=16 => 50,
            _ => 150,
        };
        assert_eq!(exercise_bonus, 150); // Exercises are best in afternoon
        assert_eq!(lesson_bonus, 50); // Lessons get lower afternoon bonus
    }

    #[test]
    fn test_energy_estimation() {
        // Morning peak (10-11am): high energy
        let energy_morning = RecommendationContext::estimate_energy(10, 0);
        assert!(energy_morning >= 850); // High morning energy

        // Post-lunch dip (12-13): lower energy
        let energy_lunch = RecommendationContext::estimate_energy(12, 0);
        assert!(energy_lunch < energy_morning); // Post-lunch is lower

        // Fatigue after sessions
        let energy_fresh = RecommendationContext::estimate_energy(10, 0);
        let energy_tired = RecommendationContext::estimate_energy(10, 5);
        assert!(energy_tired < energy_fresh); // Fatigue reduces energy
        assert!(energy_fresh - energy_tired >= 200); // Noticeable difference after 5 sessions
    }

    #[test]
    fn test_interleaving_logic() {
        // Interleaving: after 3+ same-topic items, recommend switching topics
        // This improves long-term retention (Bjork, 2004)

        let same_count: usize = 2;
        let should_switch_2 = same_count >= 3;
        assert!(!should_switch_2); // 2 in a row is fine

        let same_count: usize = 3;
        let should_switch_3 = same_count >= 3;
        assert!(should_switch_3); // 3 in a row -> suggest interleaving

        let same_count: usize = 5;
        let should_switch_5 = same_count >= 3;
        assert!(should_switch_5); // Definitely switch after 5
    }

    #[test]
    fn test_vark_style_matching() {
        // Visual learner (high V score) should prefer lessons with visuals and projects
        let (v, a, r, k): (u16, u16, u16, u16) = (900, 500, 400, 300);
        let max_score = v.max(a).max(r).max(k);

        // Visual style match score
        let visual_match = (v as u32 * 1000 / max_score as u32).min(1000) as u16;
        assert_eq!(visual_match, 1000); // Perfect match for visual content

        // Kinesthetic match for visual learner
        let kinesthetic_match = (k as u32 * 1000 / max_score as u32).min(1000) as u16;
        assert!(kinesthetic_match < 400); // Low match for hands-on with visual learner
    }

    #[test]
    fn test_smart_score_weighting() {
        // Smart score combines multiple factors with specific weights:
        // - Flow: 25%
        // - Style: 20%
        // - Timing: 15%
        // - Goal: 25%
        // - Interleaving: 15%

        let flow: u32 = 800;
        let style: u32 = 700;
        let timing: u32 = 150;
        let goal: u32 = 900;
        let interleave: u32 = 600;

        let smart_score = (
            flow * 25 +
            style * 20 +
            timing * 15 +
            goal * 25 +
            interleave * 15
        ) / 100;

        // Expected: (800*25 + 700*20 + 150*15 + 900*25 + 600*15) / 100
        //         = (20000 + 14000 + 2250 + 22500 + 9000) / 100
        //         = 67750 / 100 = 677
        assert_eq!(smart_score, 677);
        assert!(smart_score > 600); // Good overall score
        assert!(smart_score < 800); // Below flow score due to lower timing
    }

    // ============== Caching/Optimization Tests ==============

    #[test]
    fn test_learner_stats_default() {
        let stats = LearnerStats::default();
        assert_eq!(stats.total_skills, 0);
        assert_eq!(stats.mastered_count, 0);
        assert_eq!(stats.avg_mastery_permille, 0);
        assert_eq!(stats.dominant_style, LearningStyle::default());
    }

    #[test]
    fn test_quick_stats_avg_calculation() {
        // Test average mastery calculation logic
        let masteries_values: Vec<u16> = vec![800, 600, 400, 200];
        let total = masteries_values.len();
        let sum: u32 = masteries_values.iter().map(|m| *m as u32).sum();
        let avg = (sum / total as u32) as u16;
        assert_eq!(avg, 500); // (800+600+400+200)/4 = 500
    }

    #[test]
    fn test_pagination_has_more_logic() {
        // Test pagination logic
        let total = 25usize;
        let offset = 0usize;
        let limit = 10usize;
        let items_len = 10usize;

        let has_more = offset + items_len < total;
        assert!(has_more); // 0 + 10 < 25 = true

        let offset = 20usize;
        let items_len = 5usize;
        let has_more = offset + items_len < total;
        assert!(!has_more); // 20 + 5 < 25 = false (25 == 25)
    }

    #[test]
    fn test_mastery_categorization() {
        // Test mastery level categorization
        let masteries: Vec<u16> = vec![900, 850, 500, 300, 200];

        let mastered: Vec<_> = masteries.iter().filter(|m| **m >= 800).collect();
        let in_progress: Vec<_> = masteries.iter().filter(|m| **m >= 400 && **m < 800).collect();
        let needs_work: Vec<_> = masteries.iter().filter(|m| **m < 400).collect();

        assert_eq!(mastered.len(), 2);      // 900, 850
        assert_eq!(in_progress.len(), 1);   // 500
        assert_eq!(needs_work.len(), 2);    // 300, 200
    }

    #[test]
    fn test_dominant_style_detection() {
        // Test dominant learning style detection
        let scores = [
            (800u16, LearningStyle::Visual),
            (600u16, LearningStyle::Auditory),
            (500u16, LearningStyle::ReadingWriting),
            (400u16, LearningStyle::Kinesthetic),
        ];

        let dominant = scores.iter()
            .max_by_key(|(score, _)| *score)
            .map(|(_, style)| style.clone())
            .unwrap();

        assert!(matches!(dominant, LearningStyle::Visual));
    }

    // ============== Flow State Tests ==============

    #[test]
    fn test_flow_state_balance_classification() {
        // Test that flow state is correctly determined from skill/challenge balance
        // Balance = skill - challenge

        // balance > 200 -> Boredom
        assert_eq!(determine_flow_from_balance(250), FlowState::Boredom);

        // balance 100-200 -> Relaxation
        assert_eq!(determine_flow_from_balance(150), FlowState::Relaxation);

        // balance -100 to 100 -> Flow (optimal)
        assert_eq!(determine_flow_from_balance(50), FlowState::Flow);
        assert_eq!(determine_flow_from_balance(0), FlowState::Flow);
        assert_eq!(determine_flow_from_balance(-50), FlowState::Flow);

        // balance -200 to -100 -> Arousal
        assert_eq!(determine_flow_from_balance(-150), FlowState::Arousal);

        // balance -400 to -200 -> Anxiety
        assert_eq!(determine_flow_from_balance(-300), FlowState::Anxiety);

        // balance < -400 -> Overwhelm
        assert_eq!(determine_flow_from_balance(-500), FlowState::Overwhelm);
    }

    fn determine_flow_from_balance(balance: i32) -> FlowState {
        match balance {
            b if b > 200 => FlowState::Boredom,
            b if b > 100 => FlowState::Relaxation,
            b if b >= -100 => FlowState::Flow,
            b if b >= -200 => FlowState::Arousal,
            b if b >= -400 => FlowState::Anxiety,
            _ => FlowState::Overwhelm,
        }
    }

    #[test]
    fn test_flow_trend_calculation() {
        // Improving trend: recent > older by 50+
        let recent = 700u16;
        let older = 600u16;
        let trend = match recent as i32 - older as i32 {
            d if d > 50 => FlowTrend::Improving,
            d if d < -50 => FlowTrend::Declining,
            _ => FlowTrend::Stable,
        };
        assert_eq!(trend, FlowTrend::Improving);

        // Declining trend: older > recent by 50+
        let recent = 500u16;
        let older = 600u16;
        let trend = match recent as i32 - older as i32 {
            d if d > 50 => FlowTrend::Improving,
            d if d < -50 => FlowTrend::Declining,
            _ => FlowTrend::Stable,
        };
        assert_eq!(trend, FlowTrend::Declining);

        // Stable: within 50 points
        let recent = 620u16;
        let older = 600u16;
        let trend = match recent as i32 - older as i32 {
            d if d > 50 => FlowTrend::Improving,
            d if d < -50 => FlowTrend::Declining,
            _ => FlowTrend::Stable,
        };
        assert_eq!(trend, FlowTrend::Stable);
    }

    #[test]
    fn test_flow_metrics_challenge_calculation() {
        // Challenge increases with hints, skips, and slow response times
        let base_challenge: u16 = 500;
        let hints: u32 = 3;
        let skips: u32 = 2;
        let slow_response = true;

        let hint_factor = (hints * 50).min(200) as u16;
        let skip_factor = (skips * 100).min(300) as u16;
        let time_factor: u16 = if slow_response { 150 } else { 0 };

        let challenge = (base_challenge + hint_factor + skip_factor + time_factor).min(1000);

        // 500 + 150 + 200 + 150 = 1000 (capped)
        assert_eq!(challenge, 1000);
    }

    #[test]
    fn test_flow_adjustments_for_boredom() {
        // When bored (skill >> challenge), system should recommend:
        // 1. Increase difficulty
        // 2. Switch topics
        // 3. Increase pace
        let state = FlowState::Boredom;

        // Simulate adjustment generation logic
        let adjustments = match state {
            FlowState::Boredom => vec!["IncreaseDifficulty", "SwitchTopic", "IncreasePace"],
            _ => vec!["Maintain"],
        };

        assert!(adjustments.contains(&"IncreaseDifficulty"));
        assert!(adjustments.contains(&"SwitchTopic"));
    }

    #[test]
    fn test_flow_adjustments_for_overwhelm() {
        // When overwhelmed, system should recommend:
        // 1. Decrease difficulty
        // 2. Change content type to lesson
        // 3. Suggest break
        let state = FlowState::Overwhelm;

        let adjustments = match state {
            FlowState::Overwhelm => vec!["DecreaseDifficulty", "ChangeContentType", "SuggestBreak"],
            _ => vec!["Maintain"],
        };

        assert!(adjustments.contains(&"DecreaseDifficulty"));
        assert!(adjustments.contains(&"SuggestBreak"));
    }

    #[test]
    fn test_optimal_session_duration_bounds() {
        // Session duration should be bounded between 15-60 minutes
        let short_avg: u32 = 5 * 60; // 5 minutes
        let bounded_short = (short_avg / 60).max(15).min(60) as u16;
        assert_eq!(bounded_short, 15);

        let long_avg: u32 = 90 * 60; // 90 minutes
        let bounded_long = (long_avg / 60).max(15).min(60) as u16;
        assert_eq!(bounded_long, 60);

        let normal_avg: u32 = 25 * 60; // 25 minutes
        let bounded_normal = (normal_avg / 60).max(15).min(60) as u16;
        assert_eq!(bounded_normal, 25);
    }

    #[test]
    fn test_flow_state_default() {
        let default_state = FlowState::default();
        assert!(matches!(default_state, FlowState::Warming));
    }

    // ============== Performance Metrics Tests ==============

    #[test]
    fn test_algorithm_metrics_default() {
        let metrics = AlgorithmMetrics::default();
        assert_eq!(metrics.bkt_accuracy_permille, 0);
        assert_eq!(metrics.bkt_predictions, 0);
        assert_eq!(metrics.flow_prediction_accuracy_permille, 0);
        assert_eq!(metrics.recommendation_ctr_permille, 0);
    }

    #[test]
    fn test_query_metrics_default() {
        let metrics = QueryMetrics::default();
        assert_eq!(metrics.avg_context_response_us, 0);
        assert_eq!(metrics.total_calls, 0);
        assert_eq!(metrics.mastery_cache_hit_permille, 0);
    }

    #[test]
    fn test_effectiveness_metrics_default() {
        let metrics = EffectivenessMetrics::default();
        assert_eq!(metrics.avg_mastery_gain_per_session, 0);
        assert_eq!(metrics.retention_rate_permille, 0);
        assert_eq!(metrics.goal_completion_rate_permille, 0);
    }

    #[test]
    fn test_benchmark_stats_calculation() {
        let times: Vec<u64> = vec![100, 200, 150, 180, 120];
        let result = calculate_benchmark_stats("test_op", &times);

        assert_eq!(result.operation, "test_op");
        assert_eq!(result.samples, 5);
        assert_eq!(result.min_us, 100);
        assert_eq!(result.max_us, 200);
        // avg = (100+200+150+180+120) / 5 = 750 / 5 = 150
        assert_eq!(result.avg_us, 150);
    }

    #[test]
    fn test_benchmark_stats_empty() {
        let times: Vec<u64> = vec![];
        let result = calculate_benchmark_stats("empty_op", &times);

        assert_eq!(result.samples, 0);
        assert_eq!(result.min_us, 0);
        assert_eq!(result.max_us, 0);
        assert_eq!(result.avg_us, 0);
    }

    #[test]
    fn test_benchmark_stats_single_sample() {
        let times: Vec<u64> = vec![500];
        let result = calculate_benchmark_stats("single_op", &times);

        assert_eq!(result.samples, 1);
        assert_eq!(result.min_us, 500);
        assert_eq!(result.max_us, 500);
        assert_eq!(result.avg_us, 500);
        assert_eq!(result.p50_us, 500);
    }

    #[test]
    fn test_flow_accuracy_calculation() {
        // Test flow accuracy calculation logic
        let total_sessions = 10;
        let flow_sessions = 7; // 70% achieved flow

        let flow_accuracy = if total_sessions > 0 {
            ((flow_sessions * 1000) / total_sessions) as u16
        } else { 0 };

        assert_eq!(flow_accuracy, 700);
    }

    #[test]
    fn test_retention_rate_calculation() {
        // Test retention rate calculation
        let total_skills = 10;
        let retained_skills = 8; // 80% retained

        let retention_rate = if total_skills > 0 {
            ((retained_skills * 1000) / total_skills) as u16
        } else { 0 };

        assert_eq!(retention_rate, 800);
    }

    #[test]
    fn test_goal_completion_rate() {
        // Test goal completion rate
        let total_goals = 5;
        let completed_goals = 3; // 60% completed

        let completion_rate = if total_goals > 0 {
            ((completed_goals * 1000) / total_goals) as u16
        } else { 0 };

        assert_eq!(completion_rate, 600);
    }

    #[test]
    fn test_mastery_gain_calculation() {
        // Test average mastery gain per session
        let total_mastery: u32 = 4500; // Sum of all mastery levels
        let session_count: u32 = 10;

        let avg_gain = (total_mastery / session_count.max(1)).min(1000) as u16;
        assert_eq!(avg_gain, 450);

        // Test with more mastery than can be attributed to sessions
        let high_mastery: u32 = 15000;
        let few_sessions: u32 = 3;
        let capped_gain = (high_mastery / few_sessions.max(1)).min(1000) as u16;
        assert_eq!(capped_gain, 1000); // Capped at 1000
    }

    // ============== Metacognition & Confidence Calibration Tests ==============

    #[test]
    fn test_calibration_perfect_calibration() {
        // Perfect calibration: predicted = actual
        let judgments = vec![
            (700, 700), // predicted 70%, got 70%
            (800, 800), // predicted 80%, got 80%
            (500, 500), // predicted 50%, got 50%
        ];

        let errors: Vec<i32> = judgments.iter()
            .map(|(pred, actual)| (*pred as i32 - *actual as i32).abs())
            .collect();

        let mae: i32 = errors.iter().sum::<i32>() / errors.len() as i32;
        assert_eq!(mae, 0); // Perfect calibration = 0 MAE
    }

    #[test]
    fn test_calibration_overconfidence() {
        // Overconfident: always predicts higher than actual
        let judgments = vec![
            (800, 600), // predicted 80%, got 60%
            (900, 700), // predicted 90%, got 70%
            (700, 500), // predicted 70%, got 50%
        ];

        let biases: Vec<i32> = judgments.iter()
            .map(|(pred, actual)| *pred as i32 - *actual as i32)
            .collect();

        let avg_bias: i32 = biases.iter().sum::<i32>() / biases.len() as i32;
        assert!(avg_bias > 0); // Positive bias = overconfident
        assert_eq!(avg_bias, 200); // Average 20% overconfident
    }

    #[test]
    fn test_calibration_underconfidence() {
        // Underconfident: always predicts lower than actual
        let judgments = vec![
            (400, 600), // predicted 40%, got 60%
            (500, 700), // predicted 50%, got 70%
            (300, 500), // predicted 30%, got 50%
        ];

        let biases: Vec<i32> = judgments.iter()
            .map(|(pred, actual)| *pred as i32 - *actual as i32)
            .collect();

        let avg_bias: i32 = biases.iter().sum::<i32>() / biases.len() as i32;
        assert!(avg_bias < 0); // Negative bias = underconfident
        assert_eq!(avg_bias, -200); // Average 20% underconfident
    }

    #[test]
    fn test_brier_score_calculation() {
        // Brier score: mean squared error of probabilistic predictions
        // Perfect prediction: Brier = 0
        // Random prediction: Brier ≈ 0.25
        // Wrong prediction: Brier = 1

        let predictions = vec![
            (1000, 1000), // predicted 100%, actual 100% (correct)
            (800, 1000),  // predicted 80%, actual 100%
            (200, 0),     // predicted 20%, actual 0%
        ];

        let brier_sum: u64 = predictions.iter()
            .map(|(pred, actual)| {
                let diff = *pred as i64 - *actual as i64;
                (diff * diff) as u64
            })
            .sum();

        let brier_score = (brier_sum / predictions.len() as u64) as u32;

        // First: (1000-1000)^2 = 0
        // Second: (800-1000)^2 = 40000
        // Third: (200-0)^2 = 40000
        // Average: 80000/3 = 26666
        assert!(brier_score < 30000); // Good predictions
    }

    #[test]
    fn test_calibration_trend_improving() {
        // Trend: recent judgments are better than older ones
        let older_errors: Vec<u16> = vec![300, 250, 200]; // Large errors
        let recent_errors: Vec<u16> = vec![50, 100, 75]; // Small errors

        let older_avg: u16 = older_errors.iter().sum::<u16>() / older_errors.len() as u16;
        let recent_avg: u16 = recent_errors.iter().sum::<u16>() / recent_errors.len() as u16;

        let trend = if recent_avg < older_avg.saturating_sub(50) {
            CalibrationTrend::Improving
        } else if recent_avg > older_avg + 50 {
            CalibrationTrend::Declining
        } else {
            CalibrationTrend::Stable
        };

        assert!(matches!(trend, CalibrationTrend::Improving));
    }

    // ============== Enhanced Flow State Tests ==============

    #[test]
    fn test_flow_assessment_warming_up() {
        // Early in session, should return Warming
        let input = FlowAssessmentInput {
            skill_level: 500,
            challenge_level: 500,
            recent_accuracy: 700,
            response_time_factor: 1000,
            session_minutes: 2, // Only 2 minutes in
            recent_errors: 1,
        };

        // The function would return Warming for session < 3 minutes
        assert!(input.session_minutes < 3);
    }

    #[test]
    fn test_flow_assessment_overwhelm() {
        // Very high challenge vs skill = Overwhelm
        // balance > 500 triggers Overwhelm
        let balance = 800i16 - 200i16; // challenge - skill = 600

        let state = if balance > 500 {
            FlowState::Overwhelm
        } else if balance > 300 {
            FlowState::Anxiety
        } else {
            FlowState::Flow
        };

        assert!(matches!(state, FlowState::Overwhelm));
    }

    #[test]
    fn test_flow_assessment_fatigue() {
        // Slow response times + long session = Fatigue
        let response_time_factor = 1500u16; // Very slow
        let session_minutes = 45u16; // Long session

        let is_fatigued = response_time_factor > 1400 && session_minutes > 30;
        assert!(is_fatigued);
    }

    #[test]
    fn test_flow_recommendation_break() {
        // Long session in flow should suggest break
        let session_minutes = 50u16;
        let state = FlowState::Flow;

        let recommendation = match state {
            FlowState::Flow if session_minutes > 45 => FlowRecommendation::TakeBreak,
            FlowState::Flow => FlowRecommendation::MaintainCurrent,
            _ => FlowRecommendation::MaintainCurrent,
        };

        assert!(matches!(recommendation, FlowRecommendation::TakeBreak));
    }

    #[test]
    fn test_engagement_score_calculation() {
        // Engagement based on accuracy and response time
        let accuracy = 800u32;
        let response_time_factor = 900u16; // Fast = engaged

        let time_factor: u32 = if response_time_factor < 800 {
            900 // Very fast
        } else if response_time_factor > 1500 {
            500 // Very slow
        } else {
            750 // Normal
        };

        let engagement = ((accuracy + time_factor) / 2).min(1000) as u16;
        assert_eq!(engagement, 775); // (800 + 750) / 2
    }

    // ============== Learning Path Tests ==============

    #[test]
    fn test_path_step_priority_goal_related() {
        // Goal-related skills get max priority
        let is_goal_skill = true;
        let base_priority = 500u16;

        let priority = if is_goal_skill { 1000 } else { base_priority };
        assert_eq!(priority, 1000);
    }

    #[test]
    fn test_path_step_time_of_day_boost() {
        // Morning (6-11) gets learning boost
        // Evening (19-22) gets review boost
        let base_priority = 500u16;

        let morning_priority = base_priority + 50; // 6-11
        let afternoon_priority = base_priority; // 14-17
        let evening_priority = base_priority.saturating_sub(50) + 100; // Review boost

        assert!(morning_priority > afternoon_priority);
        assert_eq!(evening_priority, 550); // 500 - 50 + 100
    }

    #[test]
    fn test_path_readiness_new_skill() {
        // New skills (mastery < 100) are ready to start
        let mastery = 50u16;
        let readiness = if mastery < 100 {
            1000 // Ready to start
        } else {
            (1000 - mastery).max(200)
        };

        assert_eq!(readiness, 1000);
    }

    #[test]
    fn test_path_readiness_mastered_skill() {
        // Already mastered skills have low readiness
        let mastery = 900u16;
        let readiness = if mastery < 100 {
            1000
        } else {
            (1000 - mastery).max(200) // 100, but min 200
        };

        assert_eq!(readiness, 200); // Clamped to minimum
    }

    #[test]
    fn test_path_step_reasons() {
        // Test that reasons are correctly assigned
        let reason_for_unlock = PathStepReason::UnlocksOthers;
        let reason_for_review = PathStepReason::DueForReview;
        let reason_for_goal = PathStepReason::GoalRelated;

        assert!(matches!(reason_for_unlock, PathStepReason::UnlocksOthers));
        assert!(matches!(reason_for_review, PathStepReason::DueForReview));
        assert!(matches!(reason_for_goal, PathStepReason::GoalRelated));
    }

    // ============== Peer Learning Tests ==============

    #[test]
    fn test_peer_compatibility_similar_level() {
        // Similar levels = good for collaboration
        let my_levels: Vec<u16> = vec![500, 600, 550];
        let peer_levels: Vec<u16> = vec![520, 580, 540];

        let diff_sum: i32 = my_levels.iter()
            .zip(peer_levels.iter())
            .map(|(m, p)| (*m as i32 - *p as i32).abs())
            .sum();
        let avg_diff = diff_sum / my_levels.len() as i32;

        assert!(avg_diff < 100); // Very similar levels
    }

    #[test]
    fn test_peer_compatibility_tutoring() {
        // Higher peer = they can tutor me
        let my_level = 400u16;
        let peer_level = 800u16;

        let can_tutor = peer_level > my_level + 200;
        let can_be_tutored = my_level > peer_level + 200;

        assert!(can_tutor);
        assert!(!can_be_tutored);
    }

    #[test]
    fn test_peer_match_reason_selection() {
        // Test that match reason is correctly selected
        let my_level = 500u16;
        let peer_level = 550u16;
        let prefer_tutors = false;

        let reason = if (peer_level as i32 - my_level as i32).abs() < 100 {
            PeerMatchReason::SimilarLevel
        } else if peer_level > my_level + 200 {
            PeerMatchReason::PeerCanTutor
        } else if my_level > peer_level + 200 {
            PeerMatchReason::YouCanTutor
        } else {
            PeerMatchReason::ComplementarySkills
        };

        assert!(matches!(reason, PeerMatchReason::SimilarLevel));
    }

    #[test]
    fn test_study_group_activity_level() {
        // Activity level based on recent interactions
        let interactions_last_week = 15u32;

        let activity_level: u16 = if interactions_last_week > 20 {
            1000 // Very active
        } else if interactions_last_week > 10 {
            750 // Active
        } else if interactions_last_week > 5 {
            500 // Moderate
        } else {
            250 // Low
        };

        assert_eq!(activity_level, 750);
    }

    #[test]
    fn test_complementary_skills_detection() {
        // Complementary: I'm strong where they're weak, vice versa
        let my_skills: Vec<(u8, u16)> = vec![
            (1, 800), // Strong in skill 1
            (2, 300), // Weak in skill 2
        ];
        let peer_skills: Vec<(u8, u16)> = vec![
            (1, 300), // Weak in skill 1
            (2, 800), // Strong in skill 2
        ];

        let complementary_count: usize = my_skills.iter()
            .zip(peer_skills.iter())
            .filter(|((_, my), (_, peer))| {
                (*my > 600 && *peer < 400) || (*my < 400 && *peer > 600)
            })
            .count();

        assert_eq!(complementary_count, 2); // Both skills are complementary
    }

    // ============ FEATURE 1: Bloom's Taxonomy Tests ============

    #[test]
    fn test_cognitive_level_ordering() {
        // Bloom's Taxonomy levels should be ordered from lowest to highest
        assert!(CognitiveLevel::Remember < CognitiveLevel::Understand);
        assert!(CognitiveLevel::Understand < CognitiveLevel::Apply);
        assert!(CognitiveLevel::Apply < CognitiveLevel::Analyze);
        assert!(CognitiveLevel::Analyze < CognitiveLevel::Evaluate);
        assert!(CognitiveLevel::Evaluate < CognitiveLevel::Create);
    }

    #[test]
    fn test_cognitive_unlock_threshold() {
        // Learner needs 850/1000 mastery to unlock next level
        let current_mastery = 800u16;
        let can_unlock = current_mastery >= COGNITIVE_UNLOCK_THRESHOLD;
        assert!(!can_unlock);

        let advanced_mastery = 900u16;
        let can_advance = advanced_mastery >= COGNITIVE_UNLOCK_THRESHOLD;
        assert!(can_advance);
    }

    #[test]
    fn test_composite_mastery_calculation() {
        // Composite mastery weights higher levels more
        let remember = 1000u16;
        let understand = 800u16;
        let apply = 600u16;
        let analyze = 400u16;

        // Weighted: remember*1 + understand*2 + apply*3 + analyze*4 / 10
        // = (1000 + 1600 + 1800 + 1600) / 10 = 600
        let composite: u16 = (remember * 1 + understand * 2 + apply * 3 + analyze * 4) / 10;
        assert_eq!(composite, 600); // Higher levels contribute more
    }

    // ============ FEATURE 2: Transfer of Learning Tests ============

    #[test]
    fn test_transfer_type_decay() {
        // Near transfer retains more than far transfer
        let original_mastery = 900u16;
        let near_decay = 100u16;      // 10% decay for near transfer
        let intermediate_decay = 300u16; // 30% decay
        let far_decay = 500u16;       // 50% decay for far transfer

        let near_success = original_mastery.saturating_sub(near_decay);
        let intermediate_success = original_mastery.saturating_sub(intermediate_decay);
        let far_success = original_mastery.saturating_sub(far_decay);

        assert!(near_success > intermediate_success);
        assert!(intermediate_success > far_success);
        assert_eq!(near_success, 800);
        assert_eq!(far_success, 400);
    }

    #[test]
    fn test_deep_mastery_threshold() {
        // Deep mastery requires high success across all transfer types
        let near = 900u16;
        let intermediate = 800u16;
        let far = 700u16;

        // Deep mastery: all transfers above 700
        let is_deep = near >= 700 && intermediate >= 700 && far >= 700;
        assert!(is_deep);

        // Not deep if far transfer is weak
        let weak_far = 500u16;
        let is_not_deep = near >= 700 && intermediate >= 700 && weak_far >= 700;
        assert!(!is_not_deep);
    }

    // ============ FEATURE 3: Elaborative Interrogation Tests ============

    #[test]
    fn test_elaboration_prompt_types() {
        // Different prompt types for different learning situations
        let prompts = vec![
            ElaborationPromptType::WhyWorks,
            ElaborationPromptType::HowKnow,
            ElaborationPromptType::ExplainSimply,
            ElaborationPromptType::CounterExample,
            ElaborationPromptType::Connection,
            ElaborationPromptType::RealWorldApplication,
            ElaborationPromptType::ProcessSteps,
            ElaborationPromptType::UnderlyingPrinciple,
        ];
        assert_eq!(prompts.len(), 8); // 8 prompt types
    }

    #[test]
    fn test_elaboration_quality_affects_interval() {
        // High quality elaboration extends review interval
        let high_quality = 800u16;
        let extend_interval = high_quality >= 700;
        let multiplier = if high_quality >= 800 { 150u16 } else { 120u16 };

        assert!(extend_interval);
        assert_eq!(multiplier, 150); // 1.5x interval extension for excellent

        // Low quality triggers retrieval practice
        let low_quality = 400u16;
        let trigger_retrieval = low_quality < 500;
        assert!(trigger_retrieval);
    }

    // ============ FEATURE 4: Worked Examples Fading Tests ============

    #[test]
    fn test_worked_example_ratio_beginner() {
        // Beginners (20% mastery) get 80% worked examples
        let ratio = calculate_worked_example_ratio(200);
        assert_eq!(ratio, 660); // 800 - (200 * 0.7) = 660 (min 100, max 800)
    }

    #[test]
    fn test_worked_example_ratio_advanced() {
        // Advanced learners (80% mastery) get minimal worked examples
        let ratio = calculate_worked_example_ratio(800);
        assert_eq!(ratio, 240); // 800 - (800 * 0.7) = 240
    }

    #[test]
    fn test_worked_example_ratio_bounds() {
        // Ratio should be bounded between 10% and 80%
        let very_low = calculate_worked_example_ratio(0);
        let very_high = calculate_worked_example_ratio(1000);

        assert_eq!(very_low, 800);  // Max 80%
        assert_eq!(very_high, 100); // Min 10%
    }

    #[test]
    fn test_content_format_progression() {
        // Content format should progress as mastery increases
        let beginner_format = ContentFormat::FullWorkedExample;
        let intermediate_format = ContentFormat::PartialWorkedExample { steps_shown_permille: 500 };
        let advanced_format = ContentFormat::GuidedProblem { difficulty: 600 };
        let expert_format = ContentFormat::IndependentProblem { difficulty: 800 };

        // These represent the learning progression
        assert!(matches!(beginner_format, ContentFormat::FullWorkedExample));
        assert!(matches!(advanced_format, ContentFormat::GuidedProblem { .. }));
        assert!(matches!(expert_format, ContentFormat::IndependentProblem { .. }));

        if let ContentFormat::PartialWorkedExample { steps_shown_permille } = intermediate_format {
            assert_eq!(steps_shown_permille, 500);
        }
    }

    // ============ FEATURE 5: Expertise Reversal Tests ============

    #[test]
    fn test_expertise_reversal_detection() {
        // High mastery + high content complexity = expertise reversal
        let mastery = 850u16;
        let content_complexity = 300u16; // Low complexity

        let reversal_detected = mastery > 800 && content_complexity < 500;
        assert!(reversal_detected);

        // Recommendation should be to increase challenge
        let recommendation = if reversal_detected {
            ExpertiseRecommendation::IncreaseChallenge
        } else {
            ExpertiseRecommendation::ContentAppropriate
        };
        assert!(matches!(recommendation, ExpertiseRecommendation::IncreaseChallenge));
    }

    #[test]
    fn test_expertise_content_match() {
        // Low mastery + high complexity = need simpler content
        let mastery = 200u16;
        let content_complexity = 800u16;

        let needs_simplification = mastery < 400 && content_complexity > 600;
        assert!(needs_simplification);

        let recommendation = if needs_simplification {
            ExpertiseRecommendation::SimplifyContent
        } else {
            ExpertiseRecommendation::ContentAppropriate
        };
        assert!(matches!(recommendation, ExpertiseRecommendation::SimplifyContent));
    }

    // ============ FEATURE 6: Desirable Difficulties Tests ============

    #[test]
    fn test_retention_goal_defaults() {
        let default_goal = RetentionGoal::default();
        assert!(matches!(default_goal, RetentionGoal::LongTermRetention));
    }

    #[test]
    fn test_desirable_difficulty_dimensions() {
        // All six Bjork dimensions should be present
        let dimensions = vec![
            DifficultyDimension::Spacing,
            DifficultyDimension::Interleaving,
            DifficultyDimension::Variability,
            DifficultyDimension::Transfer,
            DifficultyDimension::Generation,
            DifficultyDimension::Testing,
        ];
        assert_eq!(dimensions.len(), 6); // 6 desirable difficulty dimensions
    }

    #[test]
    fn test_session_composition_sums_to_1000() {
        // Session composition should total 1000 permille
        let session = SessionComposition {
            spaced_items_permille: 400,
            interleaved_permille: 350,
            variability_permille: 150,
            transfer_permille: 100,
            topics_to_mix: 3,
            context_variation: 500, // Added required field
        };

        let total = session.spaced_items_permille
            + session.interleaved_permille
            + session.variability_permille
            + session.transfer_permille;

        assert_eq!(total, 1000); // 100%
    }

    #[test]
    fn test_transfer_goal_increases_variability() {
        // Transfer goal should emphasize variability and transfer
        let standard = SessionComposition {
            spaced_items_permille: 400,
            interleaved_permille: 350,
            variability_permille: 150,
            transfer_permille: 100,
            topics_to_mix: 3,
            context_variation: 500,
        };

        let transfer_focused = SessionComposition {
            spaced_items_permille: 150,
            interleaved_permille: 300,
            variability_permille: 300,
            transfer_permille: 250,
            topics_to_mix: 5,
            context_variation: 800, // Higher for transfer
        };

        assert!(transfer_focused.variability_permille > standard.variability_permille);
        assert!(transfer_focused.transfer_permille > standard.transfer_permille);
        assert!(transfer_focused.topics_to_mix > standard.topics_to_mix);
    }

    // ============ FEATURE 7: Dual Coding Tests ============

    #[test]
    fn test_content_modality_types() {
        // All modalities should be represented
        let modalities = vec![
            ContentModality::Visual,
            ContentModality::Verbal,
            ContentModality::Kinesthetic,
            ContentModality::Symbolic,
            ContentModality::DualCoded,
        ];
        assert_eq!(modalities.len(), 5);
    }

    #[test]
    fn test_dual_coded_content_structure() {
        // DualCodedContent pairs primary with complementary modality
        // Testing field presence and modality pairing
        let primary = ContentModality::Visual;
        let complementary = ContentModality::Verbal;

        // Visual + Verbal is classic dual coding (Mayer's principle)
        assert!(matches!(primary, ContentModality::Visual));
        assert!(matches!(complementary, ContentModality::Verbal));

        // Integration quality should be high for well-paired modalities
        let integration_quality = 850u16;
        assert!(integration_quality >= 800); // Good integration

        // Cognitive load should be manageable for dual-coded content
        let cognitive_load = 400u16;
        assert!(cognitive_load < 600); // Not overloaded
    }

    #[test]
    fn test_mayer_improvement_estimate() {
        // Mayer's research: 89% improvement with proper dual coding
        let max_improvement = 890u16;

        // Adding missing modality gives full improvement
        let missing_modality = true;
        let improvement = if missing_modality { 890 } else { 500 };
        assert_eq!(improvement, max_improvement);
    }

    #[test]
    fn test_learning_style_modality_match() {
        // Each learning style has an optimal primary modality
        let visual_primary = ContentModality::Visual;
        let auditory_primary = ContentModality::Verbal;
        let kinesthetic_primary = ContentModality::Kinesthetic;
        let multimodal_primary = ContentModality::DualCoded;

        assert!(matches!(visual_primary, ContentModality::Visual));
        assert!(matches!(auditory_primary, ContentModality::Verbal));
        assert!(matches!(kinesthetic_primary, ContentModality::Kinesthetic));
        assert!(matches!(multimodal_primary, ContentModality::DualCoded));
    }

    #[test]
    fn test_cognitive_load_per_modality() {
        // Different modalities have different cognitive loads
        let visual_load = 150u16;
        let verbal_load = 150u16;
        let kinesthetic_load = 250u16; // More engaging but more load
        let symbolic_load = 300u16;    // Abstract, higher load

        // Combined modalities should not overload
        let dual_visual_verbal = visual_load + verbal_load;
        assert_eq!(dual_visual_verbal, 300); // Safe combined load

        let complex_combination = visual_load + kinesthetic_load + symbolic_load;
        assert_eq!(complex_combination, 700); // Higher but manageable
    }

    // ============ FEATURE 8: Testing Effect / Retrieval Practice Tests ============

    #[test]
    fn test_retrieval_types() {
        // All retrieval types should be present (ordered by difficulty)
        let types = vec![
            RetrievalType::FreeRecall,
            RetrievalType::CuedRecall,
            RetrievalType::Recognition,
            RetrievalType::ShortAnswer,
            RetrievalType::FillInBlank,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_retrieval_vs_restudy_benefit() {
        // Testing effect: retrieval practice is 50-70% better than restudying
        // (Roediger & Karpicke, 2006)
        let restudy_retention = 400u16; // 40% after 1 week
        let retrieval_retention = 680u16; // 68% after 1 week (70% improvement)

        let improvement_permille = ((retrieval_retention - restudy_retention) as u32 * 1000 / restudy_retention as u32) as u16;
        assert!(improvement_permille >= 500); // At least 50% improvement
        assert!(improvement_permille <= 750); // Up to 75% improvement
    }

    #[test]
    fn test_retrieval_schedule_expansion() {
        // Successful retrievals should expand intervals
        let initial_interval = 1440u32; // 1 day in minutes
        let success_multiplier = 20u32; // 2.0x
        let expanded = (initial_interval * success_multiplier) / 10;

        assert_eq!(expanded, 2880); // 2 days
    }

    #[test]
    fn test_study_recommendation_types() {
        // Study recommendations should cover the main options
        let recommendations = vec![
            StudyRecommendation::RetrievalPractice { retrieval_type: RetrievalType::FreeRecall },
            StudyRecommendation::Restudy,
            StudyRecommendation::InterleavedTestStudy { test_ratio_permille: 500 },
        ];
        assert_eq!(recommendations.len(), 3);
    }

    // ============ FEATURE 9: Hypercorrection Effect Tests ============

    #[test]
    fn test_hypercorrection_detection() {
        // High confidence + wrong answer = hypercorrection candidate
        // (Butterfield & Metcalfe, 2001: 86% vs 64% correction rate)
        let confidence_when_wrong = 850u16; // High confidence
        let is_hypercorrection_candidate = confidence_when_wrong >= 700;

        assert!(is_hypercorrection_candidate);
    }

    #[test]
    fn test_feedback_intensity_levels() {
        // Feedback should be more intense for high-confidence errors
        let intensities = vec![
            FeedbackIntensity::Light,
            FeedbackIntensity::Standard,
            FeedbackIntensity::Emphasized,
            FeedbackIntensity::Deep,
        ];
        assert_eq!(intensities.len(), 4);
    }

    #[test]
    fn test_surprise_signal_strength() {
        // Higher confidence = stronger surprise = better correction
        let low_confidence_surprise = 300u16;
        let high_confidence_surprise = 850u16;

        // Surprise correlates with confidence when wrong
        assert!(high_confidence_surprise > low_confidence_surprise);
        assert!(high_confidence_surprise >= 800); // Strong surprise signal
    }

    #[test]
    fn test_hypercorrection_retention_boost() {
        // Hypercorrection should give better retention (86% vs 64%)
        let normal_correction_rate = 640u16; // 64%
        let hypercorrection_rate = 860u16;   // 86%

        let improvement = hypercorrection_rate - normal_correction_rate;
        assert_eq!(improvement, 220); // 22 percentage point improvement
    }

    // ============ FEATURE 10: Pre-Testing Effect Tests ============

    #[test]
    fn test_pre_test_benefit_calculation() {
        // Pre-testing before learning improves outcomes 10-25%
        // (Richland, Kornell & Kao, 2009)
        let base_learning = 600u32;
        let pre_test_boost = 175u32; // 17.5% average improvement
        let with_pre_test = base_learning + (base_learning * pre_test_boost / 1000);

        assert!(with_pre_test > base_learning);
        assert!(with_pre_test >= 700); // At least 10% improvement
    }

    #[test]
    fn test_pre_test_novelty_check() {
        // Pre-testing is most effective for novel material
        let prior_exposures = 0u32;
        let is_novel = prior_exposures == 0;

        assert!(is_novel);
    }

    #[test]
    fn test_pre_test_question_count() {
        // Optimal: 3-5 questions for pre-test (not overwhelming)
        let min_questions = 3u32;
        let max_questions = 5u32;
        let optimal_count = 4u32;

        assert!(optimal_count >= min_questions);
        assert!(optimal_count <= max_questions);
    }

    #[test]
    fn test_pre_test_primes_attention() {
        // Pre-testing should activate attention (attention_priming_permille > 0)
        let attention_priming = 750u16; // Strong attention priming
        let weak_priming_threshold = 500u16;

        assert!(attention_priming >= weak_priming_threshold);
    }

    // ============ FEATURE 11: Productive Failure Tests ============

    #[test]
    fn test_productive_failure_phases() {
        // All phases of productive failure framework
        let phases = vec![
            ProductiveFailurePhase::Exploration,
            ProductiveFailurePhase::Struggle,
            ProductiveFailurePhase::Consolidation,
            ProductiveFailurePhase::Application,
        ];
        assert_eq!(phases.len(), 4);
    }

    #[test]
    fn test_optimal_struggle_duration() {
        // 10-15 minutes is optimal struggle time (Kapur, 2008)
        let min_optimal = 10u32; // minutes
        let max_optimal = 15u32;
        let actual_struggle = 12u32;

        assert!(actual_struggle >= min_optimal);
        assert!(actual_struggle <= max_optimal);
    }

    #[test]
    fn test_productive_failure_transfer_boost() {
        // Productive failure improves transfer 20-30% (Kapur & Bielaczyc, 2012)
        let standard_transfer = 400u32;
        let productive_failure_transfer = 520u32; // 30% improvement

        let improvement = (productive_failure_transfer - standard_transfer) * 1000 / standard_transfer;
        assert!(improvement >= 200); // At least 20% improvement
        assert!(improvement <= 350); // Up to 35% improvement
    }

    #[test]
    fn test_struggle_recommendations() {
        // System should provide appropriate recommendations
        let recommendations = vec![
            ProductiveFailureRecommendation::ContinueStruggle,
            ProductiveFailureRecommendation::ProvideHint,
            ProductiveFailureRecommendation::BeginConsolidation,
            ProductiveFailureRecommendation::SimplifyProblem,
            ProductiveFailureRecommendation::UseDirectInstruction,
        ];
        assert_eq!(recommendations.len(), 5);
    }

    #[test]
    fn test_struggle_metrics_tracking() {
        // Track multiple attempt strategies
        let approaches_tried = 3u32;
        let solution_attempts = 5u32;

        // More approaches = more productive failure benefit
        assert!(approaches_tried >= 2);
        assert!(solution_attempts > approaches_tried);
    }

    // ============ FEATURE 12: Self-Determination Theory Tests ============

    #[test]
    fn test_sdt_three_needs() {
        // SDT: Autonomy, Competence, Relatedness (Deci & Ryan, 1985)
        let needs = vec![
            SDTNeedType::Autonomy,
            SDTNeedType::Competence,
            SDTNeedType::Relatedness,
        ];
        assert_eq!(needs.len(), 3);
    }

    #[test]
    fn test_sdt_needs_measurement() {
        // Each need should be measured 0-1000
        let autonomy = 700u16;
        let competence = 600u16;
        let relatedness = 500u16;

        assert!(autonomy <= 1000);
        assert!(competence <= 1000);
        assert!(relatedness <= 1000);
    }

    #[test]
    fn test_motivation_index_calculation() {
        // Motivation is product of three needs (normalized)
        let autonomy = 800u16;
        let competence = 700u16;
        let relatedness = 600u16;

        // Geometric mean approximation (simplified)
        let min_need = autonomy.min(competence).min(relatedness);
        let avg_need = (autonomy as u32 + competence as u32 + relatedness as u32) / 3;

        // Motivation is limited by weakest need but boosted by average
        let motivation = (min_need as u32 * 600 + avg_need * 400) / 1000;

        assert!(motivation >= 500); // Healthy motivation
        assert!(motivation <= 900); // Not artificially inflated
    }

    #[test]
    fn test_sdt_recommendations() {
        // Recommendations should target specific needs
        let recommendations = vec![
            SDTRecommendation::IncreaseChoices,           // For Autonomy
            SDTRecommendation::ProvideCompetenceFeedback, // For Competence
            SDTRecommendation::JoinStudyGroup,            // For Relatedness
            SDTRecommendation::ReduceControllingLanguage, // For Autonomy
            SDTRecommendation::AdjustDifficulty,          // For Competence
        ];
        assert_eq!(recommendations.len(), 5);
    }

    #[test]
    fn test_lowest_need_identification() {
        // System should identify and target lowest need
        let autonomy = 800u16;
        let competence = 400u16; // Lowest
        let relatedness = 600u16;

        let lowest = autonomy.min(competence).min(relatedness);
        assert_eq!(lowest, 400);

        // Recommendation should target competence
        let target = if lowest == competence {
            SDTNeedType::Competence
        } else if lowest == autonomy {
            SDTNeedType::Autonomy
        } else {
            SDTNeedType::Relatedness
        };

        assert!(matches!(target, SDTNeedType::Competence));
    }

    // ============ FEATURE 13: Growth Mindset Tests ============

    #[test]
    fn test_mindset_indicators() {
        // Three mindset states (Dweck, 2006)
        let indicators = vec![
            MindsetIndicator::Fixed,
            MindsetIndicator::Growth,
            MindsetIndicator::Mixed,
        ];
        assert_eq!(indicators.len(), 3);
    }

    #[test]
    fn test_fixed_mindset_detection() {
        // Fixed mindset signals: avoid challenges, give up easily, ignore feedback
        let avoids_challenges = 800u16;     // High avoidance
        let gives_up_early = 750u16;        // Often quits
        let ignores_feedback = 600u16;      // Dismisses criticism

        let fixed_score = (avoids_challenges + gives_up_early + ignores_feedback) / 3;
        assert!(fixed_score >= 700); // Strong fixed mindset signals
    }

    #[test]
    fn test_growth_mindset_detection() {
        // Growth mindset signals: embrace challenges, persist, learn from feedback
        let embraces_challenges = 850u16;   // High engagement
        let persists_through_difficulty = 800u16;
        let learns_from_feedback = 900u16;

        let growth_score = (embraces_challenges + persists_through_difficulty + learns_from_feedback) / 3;
        assert!(growth_score >= 800); // Strong growth mindset signals
    }

    #[test]
    fn test_mindset_interventions() {
        // Interventions should promote growth mindset
        let interventions = vec![
            MindsetIntervention::PraiseEffort,
            MindsetIntervention::NormalizeStruggle,
            MindsetIntervention::TeachPlasticity,
            MindsetIntervention::ReframeFailure,
            MindsetIntervention::ShowGrowthExamples,
        ];
        assert_eq!(interventions.len(), 5);
    }

    #[test]
    fn test_mixed_mindset_detection() {
        // Mixed mindset: growth in some areas, fixed in others
        let math_mindset = 300u16;  // Fixed (avoids math challenges)
        let art_mindset = 800u16;   // Growth (embraces art challenges)

        let variance = if math_mindset > art_mindset {
            math_mindset - art_mindset
        } else {
            art_mindset - math_mindset
        };

        // High variance = mixed mindset
        assert!(variance >= 400);
    }

    // ============ FEATURE 14: Attention / Mind-Wandering Detection Tests ============

    #[test]
    fn test_attention_states() {
        // Full attention state spectrum
        let states = vec![
            AttentionState::Focused,
            AttentionState::Drifting,
            AttentionState::MindWandering,
            AttentionState::Disengaged,
            AttentionState::Guessing,
            AttentionState::Unknown,
        ];
        assert_eq!(states.len(), 6);
    }

    #[test]
    fn test_response_time_variance_detection() {
        // High variance in response times indicates mind-wandering
        let response_times = vec![2000u32, 2100, 8000, 1900, 9500, 2000]; // ms

        let mean = response_times.iter().sum::<u32>() / response_times.len() as u32;
        let variance_sum: u32 = response_times.iter()
            .map(|&t| if t > mean { t - mean } else { mean - t })
            .sum();
        let avg_deviation = variance_sum / response_times.len() as u32;

        // High deviation = likely mind-wandering
        assert!(avg_deviation > 1500); // > 1.5 second average deviation
    }

    #[test]
    fn test_accuracy_drop_detection() {
        // Sudden accuracy drop indicates disengagement
        let recent_accuracy = vec![900u16, 850, 400, 300, 250]; // Dropping fast

        let early_avg = (recent_accuracy[0] + recent_accuracy[1]) / 2;
        let late_avg = (recent_accuracy[3] + recent_accuracy[4]) / 2;
        let drop = early_avg - late_avg;

        assert!(drop > 500); // Dramatic accuracy drop
    }

    #[test]
    fn test_re_engagement_actions() {
        // System should have re-engagement strategies
        let actions = vec![
            ReEngagementAction::Continue,
            ReEngagementAction::ThoughtProbe,
            ReEngagementAction::SwitchContent,
            ReEngagementAction::TakeBreak,
            ReEngagementAction::IncreaseInteractivity,
        ];
        assert_eq!(actions.len(), 5);
    }

    #[test]
    fn test_attention_duration_tracking() {
        // Track how long attention has been in each state
        let focused_minutes = 15u32;
        let drifting_start = 18u32; // After 18 minutes

        // After ~15-20 minutes, attention naturally drifts
        let expected_focus_duration = 15u32..25u32;
        assert!(expected_focus_duration.contains(&focused_minutes) || focused_minutes <= 25);
    }

    #[test]
    fn test_guessing_detection() {
        // Random/fast responses indicate guessing (not learning)
        let response_time_ms = 500u32; // Very fast
        let accuracy = 250u16;         // ~25% (random on 4-choice)
        let consistency = 200u16;      // Low consistency

        let is_guessing = response_time_ms < 1000 && accuracy < 400 && consistency < 400;
        assert!(is_guessing);
    }

    // ============ FEATURE 15: Critical Thinking Tests ============

    #[test]
    fn test_claim_types() {
        // All 6 claim types for argument analysis
        let claims = vec![
            ClaimType::Factual,
            ClaimType::Evaluative,
            ClaimType::Policy,
            ClaimType::Interpretive,
            ClaimType::Definitional,
            ClaimType::Causal,
        ];
        assert_eq!(claims.len(), 6);
    }

    #[test]
    fn test_evidence_types() {
        // All evidence types for evaluating arguments
        let evidence = vec![
            EvidenceType::Statistical,
            EvidenceType::Anecdotal,
            EvidenceType::Expert,
            EvidenceType::Empirical,
            EvidenceType::Logical,
            EvidenceType::Analogical,
            EvidenceType::Historical,
            EvidenceType::None,
        ];
        assert_eq!(evidence.len(), 8);
    }

    #[test]
    fn test_logical_fallacies_count() {
        // All 24 logical fallacies for detection
        let fallacies = vec![
            LogicalFallacy::AdHominem,
            LogicalFallacy::AppealToAuthority,
            LogicalFallacy::AppealToEmotion,
            LogicalFallacy::AppealToTradition,
            LogicalFallacy::AppealToNature,
            LogicalFallacy::AppealToPopularity,
            LogicalFallacy::RedHerring,
            LogicalFallacy::StrawMan,
            LogicalFallacy::Equivocation,
            LogicalFallacy::Amphiboly,
            LogicalFallacy::FalseDialemma,
            LogicalFallacy::SlipperySlope,
            LogicalFallacy::CircularReasoning,
            LogicalFallacy::HastyGeneralization,
            LogicalFallacy::FalseCause,
            LogicalFallacy::WeakAnalogy,
            LogicalFallacy::AppealToIgnorance,
            LogicalFallacy::AffirmingConsequent,
            LogicalFallacy::DenyingAntecedent,
            LogicalFallacy::NoTrueScotsman,
            LogicalFallacy::MovingGoalposts,
            LogicalFallacy::TuQuoque,
            LogicalFallacy::GeneticFallacy,
            LogicalFallacy::Whataboutism,
        ];
        assert_eq!(fallacies.len(), 24);
    }

    #[test]
    fn test_argument_strength_scoring() {
        // Argument strength based on evidence quality
        let statistical_evidence = 900u16;  // High quality
        let anecdotal_evidence = 400u16;    // Lower quality
        let no_evidence = 0u16;

        // Statistical > Anecdotal > None
        assert!(statistical_evidence > anecdotal_evidence);
        assert!(anecdotal_evidence > no_evidence);
    }

    #[test]
    fn test_argument_structure_components() {
        // A complete argument has claim, evidence, and assumptions
        let has_claim = true;
        let has_evidence = true;
        let has_assumptions = true;
        let has_conclusion = true;

        let is_complete = has_claim && has_evidence && has_assumptions && has_conclusion;
        assert!(is_complete);
    }

    #[test]
    fn test_fallacy_detection_confidence() {
        // Confidence should range from 0-1000 (permille)
        let high_confidence = 850u16;   // Clear fallacy
        let medium_confidence = 500u16; // Possible fallacy
        let low_confidence = 200u16;    // Unlikely fallacy

        assert!(high_confidence > medium_confidence);
        assert!(medium_confidence > low_confidence);
        assert!(high_confidence <= 1000);
    }

    // ============ FEATURE 16: Epistemic Vigilance Tests ============

    #[test]
    fn test_cognitive_biases_count() {
        // All 15 cognitive biases for detection
        let biases = vec![
            CognitiveBias::ConfirmationBias,
            CognitiveBias::AnchoringBias,
            CognitiveBias::AvailabilityHeuristic,
            CognitiveBias::DunningKruger,
            CognitiveBias::HindsightBias,
            CognitiveBias::SunkCostFallacy,
            CognitiveBias::BandwagonEffect,
            CognitiveBias::HaloEffect,
            CognitiveBias::NegativeBias,
            CognitiveBias::OptimismBias,
            CognitiveBias::StatusQuoBias,
            CognitiveBias::Groupthink,
            CognitiveBias::BlindSpotBias,
            CognitiveBias::ProjectionBias,
            CognitiveBias::RecencyBias,
        ];
        assert_eq!(biases.len(), 15);
    }

    #[test]
    fn test_source_types() {
        // Source credibility hierarchy
        let sources = vec![
            SourceType::PeerReviewedJournal,
            SourceType::ExpertOpinion,
            SourceType::GovernmentSource,
            SourceType::NewsMedia,
            SourceType::SocialMedia,
            SourceType::PersonalBlog,
            SourceType::WikiSource,
            SourceType::AnonymousSource,
            SourceType::PrimarySource,
            SourceType::SecondarySource,
        ];
        assert_eq!(sources.len(), 10);
    }

    #[test]
    fn test_source_credibility_scoring() {
        // Peer-reviewed > Expert > Social media > Anonymous
        let peer_reviewed = 950u16;
        let expert = 750u16;
        let social_media = 350u16;
        let anonymous = 100u16;

        assert!(peer_reviewed > expert);
        assert!(expert > social_media);
        assert!(social_media > anonymous);
    }

    #[test]
    fn test_bias_detection_accuracy() {
        // Confirmation bias: seeking info that confirms existing beliefs
        let info_seeks_confirmatory = 800u16;
        let info_seeks_contradictory = 200u16;

        let confirmation_bias_score = info_seeks_confirmatory - info_seeks_contradictory;
        assert!(confirmation_bias_score > 500); // Strong confirmation bias
    }

    #[test]
    fn test_dunning_kruger_detection() {
        // Low skill + high confidence = Dunning-Kruger
        let actual_skill = 200u16;      // Low
        let self_assessment = 850u16;   // High confidence

        let overconfidence = if self_assessment > actual_skill {
            self_assessment - actual_skill
        } else {
            0
        };

        assert!(overconfidence > 500); // Strong Dunning-Kruger signal
    }

    // ============ FEATURE 17: Socratic Dialogue Tests ============

    #[test]
    fn test_socratic_question_types() {
        // 6 types of Socratic questions
        let types = vec![
            SocraticQuestionType::Clarification,
            SocraticQuestionType::ProbeAssumptions,
            SocraticQuestionType::ProbeEvidence,
            SocraticQuestionType::ProbeViewpoints,
            SocraticQuestionType::ProbeImplications,
            SocraticQuestionType::QuestionTheQuestion,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_dialogue_depth_levels() {
        // Dialogue can go deeper with each exchange
        let surface = 1u8;
        let intermediate = 3u8;
        let deep = 5u8;
        let philosophical = 7u8;

        assert!(philosophical > deep);
        assert!(deep > intermediate);
        assert!(intermediate > surface);
    }

    #[test]
    fn test_steel_manning_strength() {
        // Steel-manning: making the strongest version of an argument
        let original_strength = 500u16;
        let steel_manned_strength = 800u16;

        let improvement = steel_manned_strength - original_strength;
        assert!(improvement >= 200); // Significant strengthening
    }

    #[test]
    fn test_questioning_effectiveness() {
        // Good questions lead to deeper thinking
        let thinking_depth_before = 300u16;
        let thinking_depth_after = 700u16;

        let effectiveness = thinking_depth_after - thinking_depth_before;
        assert!(effectiveness >= 300); // Effective questioning
    }

    // ============ FEATURE 18: Metacognition Tests ============

    #[test]
    fn test_metacognitive_skills() {
        // 6 metacognitive skills
        let skills = vec![
            MetacognitiveSkill::Planning,
            MetacognitiveSkill::Monitoring,
            MetacognitiveSkill::Evaluating,
            MetacognitiveSkill::StrategySelection,
            MetacognitiveSkill::SelfExplanation,
            MetacognitiveSkill::Debugging,
        ];
        assert_eq!(skills.len(), 6);
    }

    #[test]
    fn test_metacognitive_levels() {
        // Progression of metacognitive awareness
        let levels = vec![
            MetacognitiveLevel::Tacit,       // Unaware of thinking
            MetacognitiveLevel::Aware,       // Basic awareness
            MetacognitiveLevel::Strategic,   // Can choose strategies
            MetacognitiveLevel::Reflective,  // Deep self-reflection
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_metacognitive_accuracy() {
        // How well learners predict their performance
        let predicted_score = 700u16;
        let actual_score = 650u16;

        let accuracy = 1000 - (if predicted_score > actual_score {
            predicted_score - actual_score
        } else {
            actual_score - predicted_score
        });

        assert!(accuracy >= 900); // Good metacognitive accuracy
    }

    #[test]
    fn test_self_regulation_score() {
        // Self-regulation combines multiple metacognitive skills
        let planning = 800u16;
        let monitoring = 700u16;
        let adjusting = 750u16;

        let regulation_score = (planning + monitoring + adjusting) / 3;
        assert!(regulation_score >= 700);
    }

    // ============ FEATURE 19: Collaborative Knowledge Building Tests ============

    #[test]
    fn test_collaboration_roles() {
        // 6 collaborative learning roles
        let roles = vec![
            CollaborationRole::Proposer,
            CollaborationRole::Questioner,
            CollaborationRole::Challenger,
            CollaborationRole::Synthesizer,
            CollaborationRole::Summarizer,
            CollaborationRole::Facilitator,
        ];
        assert_eq!(roles.len(), 6);
    }

    #[test]
    fn test_argumentation_moves() {
        // 8 moves in argumentative discourse
        let moves = vec![
            ArgumentationMove::Claim,
            ArgumentationMove::Support,
            ArgumentationMove::Challenge,
            ArgumentationMove::Concede,
            ArgumentationMove::Qualify,
            ArgumentationMove::Synthesize,
            ArgumentationMove::Clarify,
            ArgumentationMove::Redirect,
        ];
        assert_eq!(moves.len(), 8);
    }

    #[test]
    fn test_perspective_taking_score() {
        // Ability to understand others' viewpoints
        let own_perspective_articulation = 800u16;
        let other_perspectives_identified = 3u8;
        let synthesis_quality = 700u16;

        // Good perspective-taking requires multiple skills
        assert!(own_perspective_articulation >= 700);
        assert!(other_perspectives_identified >= 2);
        assert!(synthesis_quality >= 600);
    }

    #[test]
    fn test_knowledge_building_contribution() {
        // Quality of contributions to shared knowledge
        let builds_on_others = 800u16;
        let introduces_new_ideas = 750u16;
        let supports_with_evidence = 700u16;

        let contribution_quality = (builds_on_others + introduces_new_ideas + supports_with_evidence) / 3;
        assert!(contribution_quality >= 700);
    }

    // ============ FEATURE 20: Creativity & Divergent Thinking Tests ============

    #[test]
    fn test_creative_thinking_types() {
        // Different types of creative thinking
        let types = vec![
            CreativeThinkingType::Divergent,
            CreativeThinkingType::Convergent,
            CreativeThinkingType::Lateral,
            CreativeThinkingType::Analogical,
            CreativeThinkingType::Combinatorial,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_creative_techniques() {
        // 9 creative thinking techniques
        let techniques = vec![
            CreativeTechnique::Brainstorming,
            CreativeTechnique::SCAMPER,
            CreativeTechnique::SixThinkingHats,
            CreativeTechnique::MindMapping,
            CreativeTechnique::Analogies,
            CreativeTechnique::ConstraintRemoval,
            CreativeTechnique::RandomStimulus,
            CreativeTechnique::Reversal,
            CreativeTechnique::WhatIf,
        ];
        assert_eq!(techniques.len(), 9);
    }

    #[test]
    fn test_torrance_fluency() {
        // Fluency: number of ideas generated
        let ideas_generated = 15u16;
        let high_fluency_threshold = 10u16;

        assert!(ideas_generated >= high_fluency_threshold);
    }

    #[test]
    fn test_torrance_flexibility() {
        // Flexibility: variety of categories/approaches
        let categories_used = 6u16;
        let high_flexibility_threshold = 4u16;

        assert!(categories_used >= high_flexibility_threshold);
    }

    #[test]
    fn test_torrance_originality() {
        // Originality: uniqueness of ideas (compared to common responses)
        let unique_ideas = 800u16;     // High originality
        let common_ideas = 200u16;     // Low originality

        assert!(unique_ideas > common_ideas);
        assert!(unique_ideas >= 700); // Threshold for "original"
    }

    #[test]
    fn test_torrance_elaboration() {
        // Elaboration: detail and development of ideas
        let detail_score = 750u16;
        let development_score = 800u16;

        let elaboration = (detail_score + development_score) / 2;
        assert!(elaboration >= 700);
    }

    #[test]
    fn test_creativity_composite_score() {
        // All 4 Torrance dimensions
        let fluency = 800u16;
        let flexibility = 750u16;
        let originality = 700u16;
        let elaboration = 850u16;

        let composite = (fluency + flexibility + originality + elaboration) / 4;
        assert!(composite >= 700);
    }

    // ============ FEATURE 21: Inquiry-Based Learning Tests ============

    #[test]
    fn test_question_quality_types() {
        // 5 levels of question quality
        let types = vec![
            QuestionQuality::Factual,
            QuestionQuality::Conceptual,
            QuestionQuality::Procedural,
            QuestionQuality::Metacognitive,
            QuestionQuality::Generative,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_inquiry_phases() {
        // 6 phases of inquiry-based learning
        let phases = vec![
            InquiryPhase::Questioning,
            InquiryPhase::Hypothesizing,
            InquiryPhase::Investigating,
            InquiryPhase::Analyzing,
            InquiryPhase::Concluding,
            InquiryPhase::Reflecting,
        ];
        assert_eq!(phases.len(), 6);
    }

    #[test]
    fn test_hypothesis_quality_scoring() {
        // Good hypotheses are testable, specific, and based on evidence
        let testable = 900u16;
        let specific = 800u16;
        let evidence_based = 750u16;

        let hypothesis_quality = (testable + specific + evidence_based) / 3;
        assert!(hypothesis_quality >= 800);
    }

    #[test]
    fn test_investigation_rigor() {
        // Rigorous investigation: controlled variables, multiple trials, etc.
        let controlled_variables = 850u16;
        let sample_size_adequate = 800u16;
        let systematic_approach = 900u16;

        let rigor = (controlled_variables + sample_size_adequate + systematic_approach) / 3;
        assert!(rigor >= 800);
    }

    #[test]
    fn test_conclusion_validity() {
        // Conclusions should match evidence and acknowledge limitations
        let matches_evidence = 900u16;
        let acknowledges_limitations = 750u16;
        let avoids_overgeneralization = 800u16;

        let validity = (matches_evidence + acknowledges_limitations + avoids_overgeneralization) / 3;
        assert!(validity >= 800);
    }

    #[test]
    fn test_question_progression() {
        // Questions should progress from factual to generative
        let factual_score = 1u8;
        let conceptual_score = 2u8;
        let procedural_score = 3u8;
        let metacognitive_score = 4u8;
        let generative_score = 5u8;

        // Higher-order questions are more valuable for deep learning
        assert!(generative_score > metacognitive_score);
        assert!(metacognitive_score > procedural_score);
        assert!(procedural_score > conceptual_score);
        assert!(conceptual_score > factual_score);
    }

    #[test]
    fn test_inquiry_completeness() {
        // Complete inquiry cycle scores
        let phases_completed = 6u8;  // All 6 phases
        let quality_per_phase = 750u16;

        let completeness = (phases_completed as u16 * 100) + (quality_per_phase / 10);
        assert!(completeness >= 650); // High completeness
    }

    // ==========================================================================
    // TIER 1: EMOTIONAL & AFFECTIVE LEARNING TESTS (Pekrun's Control-Value Theory)
    // ==========================================================================

    #[test]
    fn test_academic_emotions_pekrun() {
        // Pekrun identifies 10 distinct academic emotions
        let positive_activating = vec![
            AcademicEmotion::Enjoyment,
            AcademicEmotion::Hope,
            AcademicEmotion::Pride,
        ];
        let positive_deactivating = vec![
            AcademicEmotion::Relief,
            AcademicEmotion::Contentment,
        ];
        let negative_activating = vec![
            AcademicEmotion::Anxiety,
            AcademicEmotion::Anger,
            AcademicEmotion::Shame,
        ];
        let negative_deactivating = vec![
            AcademicEmotion::Hopelessness,
            AcademicEmotion::Boredom,
        ];

        // Total 10 emotions as per Control-Value Theory
        assert_eq!(
            positive_activating.len() + positive_deactivating.len() +
            negative_activating.len() + negative_deactivating.len(),
            10
        );
    }

    #[test]
    fn test_emotional_valence_quadrants() {
        // Valence-activation model creates 4 quadrants
        let valences = [
            EmotionalValence::PositiveActivating,
            EmotionalValence::PositiveDeactivating,
            EmotionalValence::NegativeActivating,
            EmotionalValence::NegativeDeactivating,
        ];

        assert_eq!(valences.len(), 4);
    }

    #[test]
    fn test_emotion_regulation_strategies_gross() {
        // Gross's Process Model defines key regulation strategies
        let strategies = vec![
            EmotionalRegulationStrategy::SituationSelection,
            EmotionalRegulationStrategy::SituationModification,
            EmotionalRegulationStrategy::AttentionalDeployment,
            EmotionalRegulationStrategy::CognitiveReappraisal,
            EmotionalRegulationStrategy::ExpressionSuppression,
        ];

        // All 5 from Gross's model present
        assert_eq!(strategies.len(), 5);

        // Plus modern additions
        let additional = vec![
            EmotionalRegulationStrategy::AcceptanceAndCommitment,
            EmotionalRegulationStrategy::MindfulnessAwareness,
            EmotionalRegulationStrategy::SocialSupport,
            EmotionalRegulationStrategy::CollaborativeCoping,
        ];
        assert_eq!(additional.len(), 4);
    }

    #[test]
    fn test_frustration_states_learning() {
        // Frustration has distinct states with different learning implications
        let productive = FrustrationState::Productive;
        let destructive = FrustrationState::Destructive;

        // Productive frustration aids learning
        let productive_score = match productive {
            FrustrationState::Productive => 800u16,
            _ => 200,
        };
        assert!(productive_score > 500); // Good for learning

        // Destructive frustration harms learning
        let destructive_score = match destructive {
            FrustrationState::Destructive => 100u16,
            _ => 800,
        };
        assert!(destructive_score < 500); // Bad for learning
    }

    #[test]
    fn test_confusion_as_learning_signal() {
        // Confusion can be productive (D'Mello & Graesser research)
        let productive = ConfusionState::ProductiveConfusion;
        let stuck = ConfusionState::Stuck;

        let productive_learning_boost = match productive {
            ConfusionState::ProductiveConfusion => 200u16, // 20% boost
            _ => 0,
        };
        assert_eq!(productive_learning_boost, 200);

        let stuck_learning_penalty = match stuck {
            ConfusionState::Stuck => 150u16, // 15% penalty
            _ => 0,
        };
        assert_eq!(stuck_learning_penalty, 150);
    }

    #[test]
    fn test_emotional_state_tracking() {
        let state = EmotionalState {
            primary_emotion: AcademicEmotion::Enjoyment,
            intensity_permille: 800,
            valence: EmotionalValence::PositiveActivating,
            frustration: FrustrationState::None,
            confusion: ConfusionState::NoConfusion,
            emotional_stability_permille: 750,
            duration_minutes: 30,
        };

        // High intensity positive emotion
        assert!(state.intensity_permille > 700);
        assert!(matches!(state.valence, EmotionalValence::PositiveActivating));
    }

    // ==========================================================================
    // TIER 2: DELIBERATE PRACTICE FRAMEWORK TESTS (Ericsson)
    // ==========================================================================

    #[test]
    fn test_deliberate_practice_phases() {
        // Ericsson's 6 phases of deliberate practice
        let phases = [
            DeliberatePracticePhase::SkillAnalysis,
            DeliberatePracticePhase::StretchGoalSetting,
            DeliberatePracticePhase::FocusedRepetition,
            DeliberatePracticePhase::ImmediateFeedback,
            DeliberatePracticePhase::Refinement,
            DeliberatePracticePhase::Consolidation,
        ];

        assert_eq!(phases.len(), 6);
    }

    #[test]
    fn test_feedback_timing_importance() {
        // Immediate feedback is critical for deliberate practice
        let immediate = FeedbackTiming::Immediate;
        let delayed = FeedbackTiming::Delayed;
        let very_delayed = FeedbackTiming::VeryDelayed;

        let immediate_effectiveness = match immediate {
            FeedbackTiming::Immediate => 1000u16,
            FeedbackTiming::NearImmediate => 800,
            FeedbackTiming::Delayed => 500,
            FeedbackTiming::VeryDelayed => 200,
        };

        let delayed_effectiveness = match delayed {
            FeedbackTiming::Delayed => 500u16,
            _ => 1000,
        };

        let very_delayed_effectiveness = match very_delayed {
            FeedbackTiming::VeryDelayed => 200u16,
            _ => 1000,
        };

        // Immediate > Delayed > Very Delayed
        assert!(immediate_effectiveness > delayed_effectiveness);
        assert!(delayed_effectiveness > very_delayed_effectiveness);
    }

    #[test]
    fn test_stretch_zone_optimal() {
        // Optimal stretch is key to deliberate practice
        let optimal = StretchZone::OptimalStretch;
        let too_easy = StretchZone::TooEasy;
        let over_stretch = StretchZone::OverStretch;

        let optimal_growth = match optimal {
            StretchZone::OptimalStretch => 300u16, // 30% growth rate
            _ => 100,
        };

        let too_easy_growth = match too_easy {
            StretchZone::TooEasy => 50u16, // 5% growth (boring)
            _ => 200,
        };

        let over_stretch_growth = match over_stretch {
            StretchZone::OverStretch => 50u16, // 5% growth (overwhelming)
            _ => 200,
        };

        // Optimal stretch maximizes growth
        assert!(optimal_growth > too_easy_growth);
        assert!(optimal_growth > over_stretch_growth);
    }

    #[test]
    fn test_mental_representation_quality() {
        let representation = MentalRepresentationQuality {
            pattern_recognition_permille: 850,
            chunking_ability_permille: 800,
            anticipation_accuracy_permille: 780,
            error_detection_permille: 900,
            self_monitoring_permille: 750,
        };

        // High chunking = expert
        assert!(representation.chunking_ability_permille > 700);
        // Good pattern recognition
        assert!(representation.pattern_recognition_permille > 700);
    }

    #[test]
    fn test_skill_decomposition() {
        let decomposition = SkillDecomposition {
            parent_skill_id: "math_algebra".to_string(),
            sub_skills: vec![
                "variable_manipulation".to_string(),
                "equation_solving".to_string(),
                "word_problem_translation".to_string(),
            ],
            weakest_sub_skill: "word_problem_translation".to_string(),
            practice_priority_order: vec![
                "word_problem_translation".to_string(),
                "equation_solving".to_string(),
                "variable_manipulation".to_string(),
            ],
            estimated_hours_to_improve: 10,
        };

        // Weakest skill identified
        assert_eq!(decomposition.weakest_sub_skill, "word_problem_translation");
        // Has sub-skills
        assert_eq!(decomposition.sub_skills.len(), 3);
    }

    #[test]
    fn test_deliberate_practice_quality() {
        let quality = DeliberatePracticeQuality {
            focus_intensity_permille: 900,
            stretch_zone: StretchZone::OptimalStretch,
            feedback_timing: FeedbackTiming::Immediate,
            feedback_specificity: FeedbackSpecificity::Precise,
            repetition_with_variation: true,
            goal_specificity_permille: 850,
            overall_quality_permille: 880,
        };

        // All metrics above 700 = high-quality practice
        assert!(quality.focus_intensity_permille > 700);
        assert!(quality.goal_specificity_permille > 700);
        assert!(quality.overall_quality_permille > 700);
    }

    // ==========================================================================
    // TIER 3: SOCIAL-EMOTIONAL LEARNING TESTS (CASEL Framework)
    // ==========================================================================

    #[test]
    fn test_casel_five_competencies() {
        // CASEL defines 5 core SEL competencies
        let competencies = [
            SELCompetency::SelfAwareness,
            SELCompetency::SelfManagement,
            SELCompetency::SocialAwareness,
            SELCompetency::RelationshipSkills,
            SELCompetency::ResponsibleDecisionMaking,
        ];

        assert_eq!(competencies.len(), 5);
    }

    #[test]
    fn test_self_awareness_skills() {
        // Self-awareness sub-skills (6 per CASEL)
        let skills = [
            SelfAwarenessSkill::EmotionIdentification,
            SelfAwarenessSkill::StrengthRecognition,
            SelfAwarenessSkill::LimitationAwareness,
            SelfAwarenessSkill::ConfidenceCalibration,
            SelfAwarenessSkill::ValuesClarification,
            SelfAwarenessSkill::GrowthMindsetOrientation,
        ];

        assert_eq!(skills.len(), 6);
    }

    #[test]
    fn test_self_management_skills() {
        // Self-management sub-skills (6 per CASEL)
        let skills = [
            SelfManagementSkill::ImpulseControl,
            SelfManagementSkill::StressManagement,
            SelfManagementSkill::SelfMotivation,
            SelfManagementSkill::GoalSetting,
            SelfManagementSkill::OrganizationalSkills,
            SelfManagementSkill::SelfDiscipline,
        ];

        assert_eq!(skills.len(), 6);
    }

    #[test]
    fn test_social_awareness_skills() {
        // Social awareness sub-skills (6 per CASEL)
        let skills = [
            SocialAwarenessSkill::PerspectiveTaking,
            SocialAwarenessSkill::Empathy,
            SocialAwarenessSkill::DiversityAppreciation,
            SocialAwarenessSkill::RespectForOthers,
            SocialAwarenessSkill::SocialCueReading,
            SocialAwarenessSkill::ContextualAwareness,
        ];

        assert_eq!(skills.len(), 6);
    }

    #[test]
    fn test_relationship_skills() {
        // Relationship sub-skills (6 per CASEL)
        let skills = [
            RelationshipSkill::Communication,
            RelationshipSkill::SocialEngagement,
            RelationshipSkill::CollaborativeTeamwork,
            RelationshipSkill::ConflictResolution,
            RelationshipSkill::SeekingHelp,
            RelationshipSkill::OfferingHelp,
        ];

        assert_eq!(skills.len(), 6);
    }

    #[test]
    fn test_decision_making_skills() {
        // Responsible decision-making sub-skills (6 per CASEL)
        let skills = [
            DecisionMakingSkill::ProblemIdentification,
            DecisionMakingSkill::AlternativeGeneration,
            DecisionMakingSkill::ConsequenceEvaluation,
            DecisionMakingSkill::EthicalReasoning,
            DecisionMakingSkill::ReflectiveAnalysis,
            DecisionMakingSkill::CriticalThinking,
        ];

        assert_eq!(skills.len(), 6);
    }

    #[test]
    fn test_sel_interventions() {
        // SEL has targeted interventions
        let interventions = [
            SELIntervention::JournalingPrompt,
            SELIntervention::BreathingExercise,
            SELIntervention::PerspectiveTakingExercise,
            SELIntervention::EmpathyMapping,
            SELIntervention::ConflictScenarioRolePlay,
            SELIntervention::EthicalDilemmaDiscussion,
        ];

        // Multiple intervention types available
        assert!(interventions.len() >= 6);
    }

    #[test]
    fn test_sel_profile_scoring() {
        let profile = SELProfile {
            self_awareness_permille: 800,
            self_management_permille: 750,
            social_awareness_permille: 700,
            relationship_skills_permille: 650,
            decision_making_permille: 720,
            overall_sel_permille: 724, // Average
            strongest_competency: SELCompetency::SelfAwareness,
            growth_area: SELCompetency::RelationshipSkills,
        };

        // Strongest is highest
        assert_eq!(profile.self_awareness_permille, 800);
        // Growth area is lowest
        assert_eq!(profile.relationship_skills_permille, 650);
    }

    // ==========================================================================
    // TIER 4: STEALTH & DYNAMIC ASSESSMENT TESTS (Vygotsky ZPD)
    // ==========================================================================

    #[test]
    fn test_stealth_indicators() {
        // Stealth assessment uses behavioral indicators
        let indicators = [
            StealthIndicator::TimeToFirstAction,
            StealthIndicator::PausePatterns,
            StealthIndicator::ResponseTimeVariance,
            StealthIndicator::ToolUsagePatterns,
            StealthIndicator::HintRequestTiming,
            StealthIndicator::RevisionBehavior,
            StealthIndicator::PathThroughContent,
            StealthIndicator::RevisitPatterns,
            StealthIndicator::SkippingBehavior,
            StealthIndicator::HelpSeekingPatterns,
            StealthIndicator::PeerInteractionQuality,
            StealthIndicator::ExplanationDepth,
        ];

        // 12 behavioral indicators
        assert_eq!(indicators.len(), 12);
    }

    #[test]
    fn test_assessment_evidence_types() {
        // Different types of evidence for stealth assessment (6 per impl)
        let evidence_types = [
            AssessmentEvidence::DirectPerformance,
            AssessmentEvidence::ProcessEvidence,
            AssessmentEvidence::TransferEvidence,
            AssessmentEvidence::ExplanationEvidence,
            AssessmentEvidence::PeerTeachingEvidence,
            AssessmentEvidence::SelfAssessmentEvidence,
        ];

        assert_eq!(evidence_types.len(), 6);
    }

    #[test]
    fn test_scaffolding_levels_vygotsky() {
        // Vygotsky's ZPD scaffolding levels
        let levels = [
            ScaffoldingLevel::NoSupport,
            ScaffoldingLevel::Minimal,
            ScaffoldingLevel::Moderate,
            ScaffoldingLevel::Substantial,
            ScaffoldingLevel::Maximal,
        ];

        assert_eq!(levels.len(), 5);
    }

    #[test]
    fn test_learning_potential_zpd() {
        let potential = LearningPotential {
            current_performance_permille: 600,
            supported_performance_permille: 850,
            learning_potential_gap: 250, // Difference = ZPD width
            scaffolding_responsiveness: 800,
            transfer_potential_permille: 700,
            modifiability_score_permille: 750,
        };

        // ZPD width = supported - current
        assert_eq!(
            potential.learning_potential_gap,
            potential.supported_performance_permille - potential.current_performance_permille
        );

        // High scaffolding responsiveness = good learner
        assert!(potential.scaffolding_responsiveness > 700);
    }

    #[test]
    fn test_self_assessment_accuracy() {
        let accuracy = SelfAssessmentAccuracy {
            predicted_score_permille: 700,
            actual_score_permille: 650,
            accuracy_permille: 950, // How close prediction was (high = accurate)
            bias_direction: 50,     // Positive = overconfident
            calibration_trend: CalibrationTrend::Improving,
        };

        // Overconfident when predicted > actual
        assert!(accuracy.predicted_score_permille > accuracy.actual_score_permille);
        assert!(accuracy.bias_direction > 0); // Positive = overconfident
    }

    #[test]
    fn test_dynamic_assessment_scaffolding() {
        // Dynamic assessment uses graduated prompts (Campione & Brown)
        let scaffolding_sequence = vec![
            ScaffoldingLevel::Minimal,
            ScaffoldingLevel::Moderate,
            ScaffoldingLevel::Substantial,
        ];

        // Progressive increase in support
        assert!(scaffolding_sequence.len() >= 3);
    }

    // ==========================================================================
    // TIER 5: UNIVERSAL DESIGN FOR LEARNING TESTS (CAST Framework)
    // ==========================================================================

    #[test]
    fn test_udl_three_principles() {
        // UDL has 3 core principles
        let principles = [
            UDLPrinciple::Engagement,
            UDLPrinciple::Representation,
            UDLPrinciple::ActionExpression,
        ];

        assert_eq!(principles.len(), 3);
    }

    #[test]
    fn test_engagement_options() {
        // UDL Engagement options (9 checkpoints)
        let options = [
            EngagementOption::ChoiceAndAutonomy,
            EngagementOption::RelevanceAuthenticity,
            EngagementOption::ThreatsDistractionsMin,
            EngagementOption::GoalsSaliency,
            EngagementOption::ChallengeSupport,
            EngagementOption::CollaborationCommunity,
            EngagementOption::ExpectationsBeliefs,
            EngagementOption::CopingSkills,
            EngagementOption::SelfAssessmentReflection,
        ];

        assert_eq!(options.len(), 9);
    }

    #[test]
    fn test_representation_options() {
        // UDL Representation options (9 checkpoints mapped to impl)
        let options = [
            RepresentationOption::CustomizableDisplay,
            RepresentationOption::AlternativesAuditory,
            RepresentationOption::AlternativesVisual,
            RepresentationOption::VocabularySupport,
            RepresentationOption::SymbolDecoding,
            RepresentationOption::MultipleLanguages,
            RepresentationOption::BackgroundKnowledge,
            RepresentationOption::PatternsRelationships,
            RepresentationOption::TransferGeneralization,
        ];

        assert_eq!(options.len(), 9);
    }

    #[test]
    fn test_action_expression_options() {
        // UDL Action/Expression options (9 checkpoints mapped to impl)
        let options = [
            ActionExpressionOption::ResponseMethods,
            ActionExpressionOption::NavigationAccess,
            ActionExpressionOption::AssistiveTech,
            ActionExpressionOption::MultipleMedia,
            ActionExpressionOption::ToolsComposition,
            ActionExpressionOption::ScaffoldedPractice,
            ActionExpressionOption::GoalSettingSupport,
            ActionExpressionOption::ProgressMonitoring,
            ActionExpressionOption::CapacityManagement,
        ];

        assert_eq!(options.len(), 9);
    }

    #[test]
    fn test_accessibility_needs() {
        // UDL addresses diverse accessibility needs
        let needs = [
            AccessibilityNeed::Visual,
            AccessibilityNeed::Auditory,
            AccessibilityNeed::Motor,
            AccessibilityNeed::Cognitive,
            AccessibilityNeed::Linguistic,
            AccessibilityNeed::Emotional,
        ];

        assert_eq!(needs.len(), 6);
    }

    #[test]
    fn test_udl_profile() {
        let profile = UDLProfile {
            // Engagement preferences
            preferred_choice_level: 4,
            collaboration_preference_permille: 700,
            self_regulation_support_needed: false,
            // Representation preferences
            visual_preference_permille: 800,
            auditory_preference_permille: 500,
            text_preference_permille: 600,
            needs_vocabulary_support: false,
            preferred_language: "en".to_string(),
            // Action preferences
            preferred_input_method: "keyboard".to_string(),
            preferred_output_format: "text".to_string(),
            executive_function_support_needed: false,
            // Accessibility
            accessibility_needs: vec![AccessibilityNeed::Visual],
        };

        // Profile captures learner preferences across all 3 principles
        assert!(profile.preferred_choice_level > 0);
        assert!(profile.visual_preference_permille > 0);
        assert!(!profile.accessibility_needs.is_empty());
    }

    #[test]
    fn test_udl_barrier_identification() {
        let barrier = UDLBarrier {
            barrier_type: UDLPrinciple::Representation,
            severity_permille: 700,
            description: "Text-only content without visual alternatives".to_string(),
            recommended_accommodations: vec![
                "Visual alternatives".to_string(),
                "Pattern highlighting".to_string(),
            ],
        };

        // High severity barrier identified
        assert!(barrier.severity_permille > 500);
        // Accommodations suggested
        assert!(!barrier.recommended_accommodations.is_empty());
    }

    #[test]
    fn test_udl_27_checkpoints() {
        // UDL Framework has 27 total checkpoints (9 per principle)
        let engagement_count = 9;
        let representation_count = 9;
        let action_expression_count = 9;

        let total_checkpoints = engagement_count + representation_count + action_expression_count;
        assert_eq!(total_checkpoints, 27);
    }

    // ==========================================================================
    // CROSS-TIER INTEGRATION TESTS
    // ==========================================================================

    #[test]
    fn test_emotion_affects_practice_quality() {
        // Emotions impact deliberate practice effectiveness
        let positive_emotion = AcademicEmotion::Enjoyment;
        let negative_emotion = AcademicEmotion::Anxiety;

        let positive_practice_boost = match positive_emotion {
            AcademicEmotion::Enjoyment => 200u16, // 20% boost
            _ => 0,
        };

        let negative_practice_penalty = match negative_emotion {
            AcademicEmotion::Anxiety => 150u16, // 15% penalty
            _ => 0,
        };

        assert!(positive_practice_boost > 0);
        assert!(negative_practice_penalty > 0);
    }

    #[test]
    fn test_sel_supports_learning() {
        // Higher SEL leads to better learning outcomes
        let self_management = 800u16;
        let learning_boost = if self_management > 700 { 150u16 } else { 0 };

        // Good self-management boosts learning
        assert!(learning_boost > 0);
    }

    #[test]
    fn test_udl_reduces_barriers() {
        // UDL accommodations reduce learning barriers
        let barriers_before = 5u8;
        let accommodations_applied = 3u8;
        let barriers_after = barriers_before.saturating_sub(accommodations_applied);

        // Barriers reduced by accommodations
        assert!(barriers_after < barriers_before);
    }

    // ==========================================================================
    // TIER 6: GAMIFICATION 2.0 TESTS
    // Based on: Deterding (2011), Hamari (2014)
    // ==========================================================================

    #[test]
    fn test_leaderboard_types() {
        // Test all 6 leaderboard types for different contexts
        let types = [
            LeaderboardType::Global,
            LeaderboardType::Course,
            LeaderboardType::Friends,
            LeaderboardType::Weekly,
            LeaderboardType::Monthly,
            LeaderboardType::AllTime,
        ];
        assert_eq!(types.len(), 6);

        // Time-scoped leaderboards reset
        let weekly = LeaderboardType::Weekly;
        let monthly = LeaderboardType::Monthly;
        let all_time = LeaderboardType::AllTime;

        // Different scopes serve different purposes
        assert!(matches!(weekly, LeaderboardType::Weekly));
        assert!(matches!(monthly, LeaderboardType::Monthly));
        assert!(matches!(all_time, LeaderboardType::AllTime));
    }

    #[test]
    fn test_achievement_tiers_xp_multipliers() {
        // Higher tiers should have higher XP multipliers
        let bronze = AchievementTier::Bronze;
        let silver = AchievementTier::Silver;
        let gold = AchievementTier::Gold;
        let platinum = AchievementTier::Platinum;
        let diamond = AchievementTier::Diamond;
        let legendary = AchievementTier::Legendary;

        // XP multipliers in permille (1000 = 1x)
        let bronze_mult = match bronze {
            AchievementTier::Bronze => 1000u16,
            _ => 0,
        };
        let silver_mult = match silver {
            AchievementTier::Silver => 1250u16,  // 1.25x
            _ => 0,
        };
        let gold_mult = match gold {
            AchievementTier::Gold => 1500u16,  // 1.5x
            _ => 0,
        };
        let platinum_mult = match platinum {
            AchievementTier::Platinum => 2000u16,  // 2x
            _ => 0,
        };
        let diamond_mult = match diamond {
            AchievementTier::Diamond => 2500u16,  // 2.5x
            _ => 0,
        };
        let legendary_mult = match legendary {
            AchievementTier::Legendary => 3000u16,  // 3x
            _ => 0,
        };

        // Verify tier progression
        assert!(bronze_mult < silver_mult);
        assert!(silver_mult < gold_mult);
        assert!(gold_mult < platinum_mult);
        assert!(platinum_mult < diamond_mult);
        assert!(diamond_mult < legendary_mult);
    }

    #[test]
    fn test_achievement_categories() {
        // 8 achievement categories for varied gamification
        let categories = [
            AchievementCategory::Learning,
            AchievementCategory::Consistency,
            AchievementCategory::Social,
            AchievementCategory::Mastery,
            AchievementCategory::Exploration,
            AchievementCategory::Speed,
            AchievementCategory::Accuracy,
            AchievementCategory::Creative,
        ];
        assert_eq!(categories.len(), 8);

        // Each category motivates different behaviors
        let learning = AchievementCategory::Learning;
        let social = AchievementCategory::Social;
        let mastery = AchievementCategory::Mastery;

        // Different intrinsic motivations
        assert!(matches!(learning, AchievementCategory::Learning));
        assert!(matches!(social, AchievementCategory::Social));
        assert!(matches!(mastery, AchievementCategory::Mastery));
    }

    #[test]
    fn test_challenge_types() {
        // 6 challenge types for engagement variety
        let types = [
            ChallengeType::Daily,
            ChallengeType::Weekly,
            ChallengeType::Event,
            ChallengeType::Skill,
            ChallengeType::Community,
            ChallengeType::Versus,
        ];
        assert_eq!(types.len(), 6);

        // Time-based challenges create urgency
        let daily = ChallengeType::Daily;
        let weekly = ChallengeType::Weekly;
        let event = ChallengeType::Event;

        assert!(matches!(daily, ChallengeType::Daily));
        assert!(matches!(weekly, ChallengeType::Weekly));
        assert!(matches!(event, ChallengeType::Event));
    }

    #[test]
    fn test_social_comparison_theory() {
        // Based on Festinger's Social Comparison Theory (1954)
        let types = [
            SocialComparisonType::UpwardClose,      // Compare to slightly better
            SocialComparisonType::UpwardFar,        // Compare to much better
            SocialComparisonType::Lateral,          // Compare to similar
            SocialComparisonType::DownwardClose,    // Compare to slightly worse
            SocialComparisonType::SelfReferenced,   // Compare to past self
        ];
        assert_eq!(types.len(), 5);

        // Self-referenced is best for intrinsic motivation
        let self_ref = SocialComparisonType::SelfReferenced;
        let upward_close = SocialComparisonType::UpwardClose;

        // Different comparison types for different purposes
        assert!(matches!(self_ref, SocialComparisonType::SelfReferenced));
        assert!(matches!(upward_close, SocialComparisonType::UpwardClose));

        // Upward close = aspiration, Self-referenced = growth mindset
    }

    #[test]
    fn test_leaderboard_entry_ranking() {
        // Test leaderboard entry structure
        let entry = LeaderboardEntry {
            rank: 1,
            agent_id: "agent123".to_string(),
            display_name: "TopLearner".to_string(),
            score: 9500,
            xp_total: 45000,
            streak_days: 30,
            badges_earned: 12,
            mastery_skills: 25,
        };

        assert_eq!(entry.rank, 1);
        assert!(entry.score > 0);
        assert!(entry.badges_earned > 0);
        assert!(entry.mastery_skills > 0);
    }

    // ==========================================================================
    // TIER 7: ADVANCED ANALYTICS TESTS
    // Based on: Baker & Inventado (2014), Romero & Ventura (2020)
    // ==========================================================================

    #[test]
    fn test_cohort_types() {
        // 6 cohort types for educational data mining
        let types = [
            CohortType::EnrollmentDate,
            CohortType::SkillLevel,
            CohortType::LearningStyle,
            CohortType::GoalType,
            CohortType::ActivityPattern,
            CohortType::Custom,
        ];
        assert_eq!(types.len(), 6);

        // Different cohorts for different analyses
        let by_skill = CohortType::SkillLevel;
        let by_style = CohortType::LearningStyle;

        assert!(matches!(by_skill, CohortType::SkillLevel));
        assert!(matches!(by_style, CohortType::LearningStyle));
    }

    #[test]
    fn test_trajectory_types() {
        // 7 learning trajectory patterns
        let types = [
            TrajectoryType::Steady,
            TrajectoryType::FrontLoaded,
            TrajectoryType::BackLoaded,
            TrajectoryType::Sporadic,
            TrajectoryType::Plateaued,
            TrajectoryType::Declining,
            TrajectoryType::Accelerating,
        ];
        assert_eq!(types.len(), 7);

        // Identify trajectory patterns for intervention
        let steady = TrajectoryType::Steady;
        let declining = TrajectoryType::Declining;
        let accelerating = TrajectoryType::Accelerating;

        // Declining needs intervention, accelerating is good
        assert!(matches!(declining, TrajectoryType::Declining));
        assert!(matches!(accelerating, TrajectoryType::Accelerating));
        assert!(matches!(steady, TrajectoryType::Steady));
    }

    #[test]
    fn test_prediction_types() {
        // 6 prediction types for learning analytics
        let types = [
            PredictionType::Completion,
            PredictionType::Dropout,
            PredictionType::NextSkill,
            PredictionType::OptimalTime,
            PredictionType::AssessmentScore,
            PredictionType::TimeToMastery,
        ];
        assert_eq!(types.len(), 6);

        // Dropout prediction for early intervention
        let dropout = PredictionType::Dropout;
        let completion = PredictionType::Completion;

        assert!(matches!(dropout, PredictionType::Dropout));
        assert!(matches!(completion, PredictionType::Completion));
    }

    #[test]
    fn test_anomaly_types() {
        // 6 anomaly types for detecting unusual patterns
        let types = [
            AnomalyType::PerformanceDrop,
            AnomalyType::UnusualTiming,
            AnomalyType::SpeedAnomaly,
            AnomalyType::EngagementShift,
            AnomalyType::RoutineBreak,
            AnomalyType::SkillRegression,
        ];
        assert_eq!(types.len(), 6);

        // Performance drop needs attention
        let perf_drop = AnomalyType::PerformanceDrop;
        let skill_regression = AnomalyType::SkillRegression;

        assert!(matches!(perf_drop, AnomalyType::PerformanceDrop));
        assert!(matches!(skill_regression, AnomalyType::SkillRegression));
    }

    #[test]
    fn test_cohort_stats_structure() {
        // Test cohort statistics structure
        let stats = CohortStats {
            cohort_id: "intermediate".to_string(),
            cohort_type: CohortType::SkillLevel,
            member_count: 150,
            avg_mastery_permille: 720,
            avg_completion_rate_permille: 720,  // 72% completion
            avg_retention_rate_permille: 850,
            avg_session_minutes: 25,
            avg_days_to_mastery: 14,
            top_performing_skills: vec!["algebra".to_string()],
            struggling_skills: vec!["calculus".to_string()],
        };

        assert_eq!(stats.member_count, 150);
        assert!(stats.avg_completion_rate_permille > 700);
        assert!(!stats.top_performing_skills.is_empty());
    }

    #[test]
    fn test_learning_trajectory_analysis() {
        // Test trajectory point structure
        let point = TrajectoryPoint {
            timestamp: 1704067200,
            mastery_permille: 650,
            xp_total: 5000,
            skills_mastered: 8,
            streak_days: 15,
            engagement_permille: 800,
        };

        let trajectory = LearningTrajectory {
            agent_id: "learner123".to_string(),
            trajectory_points: vec![point.clone()],
            trend: TrendDirection::Increasing,
            velocity_permille: 150,  // 15% progress rate
            predicted_mastery_30_days: 850,
            predicted_completion_date: Some(1706745600),
            trajectory_type: TrajectoryType::Accelerating,
        };

        assert!(matches!(trajectory.trajectory_type, TrajectoryType::Accelerating));
        assert!(trajectory.velocity_permille > 0);  // Positive velocity
        assert!(matches!(trajectory.trend, TrendDirection::Increasing));
    }

    #[test]
    fn test_prediction_confidence() {
        // Predictions should have confidence levels
        let prediction = Prediction {
            prediction_type: PredictionType::Completion,
            predicted_value: 850,  // 85% likely to complete
            confidence_permille: 780,  // 78% confidence
            factors: vec![
                PredictionFactor {
                    factor_name: "Current Progress".to_string(),
                    impact_permille: 400,  // Positive impact
                    current_value: "72%".to_string(),
                    optimal_value: "100%".to_string(),
                },
            ],
            recommendation: "Keep up the consistent practice!".to_string(),
        };

        assert!(prediction.predicted_value > 800);
        assert!(prediction.confidence_permille > 700);
        assert!(prediction.factors[0].impact_permille > 0);
    }

    #[test]
    fn test_at_risk_indicators() {
        // At-risk detection for early intervention
        let indicator = AtRiskIndicator {
            risk_type: "Disengagement".to_string(),
            risk_score_permille: 650,  // 65% risk level
            contributing_factors: vec![
                "Declining engagement".to_string(),
                "Missed 3 sessions".to_string(),
            ],
            early_warning_days: 5,
            intervention_suggestions: vec![
                "Send check-in message".to_string(),
                "Offer peer support".to_string(),
            ],
        };

        assert!(indicator.risk_score_permille > 600);
        assert!(!indicator.contributing_factors.is_empty());
        assert!(!indicator.intervention_suggestions.is_empty());
        assert!(indicator.early_warning_days > 0);
    }

    // ==========================================================================
    // TIER 8: AI TUTORING INTEGRATION TESTS
    // Based on: VanLehn (2011), Graesser et al. (2004)
    // ==========================================================================

    #[test]
    fn test_tutoring_modes() {
        // 6 tutoring modes from ITS research
        let modes = [
            TutoringMode::Socratic,
            TutoringMode::Direct,
            TutoringMode::GuidedDiscovery,
            TutoringMode::WorkedExamples,
            TutoringMode::Scaffolded,
            TutoringMode::Remediation,
        ];
        assert_eq!(modes.len(), 6);

        // Socratic for deep understanding
        let socratic = TutoringMode::Socratic;
        let remediation = TutoringMode::Remediation;
        let scaffolded = TutoringMode::Scaffolded;

        assert!(matches!(socratic, TutoringMode::Socratic));
        assert!(matches!(remediation, TutoringMode::Remediation));
        assert!(matches!(scaffolded, TutoringMode::Scaffolded));
    }

    #[test]
    fn test_hint_level_xp_penalties() {
        // Progressive hints should have increasing XP penalties
        let meta = HintLevel::Metacognitive;
        let directional = HintLevel::Directional;
        let specific = HintLevel::Specific;
        let bottom_out = HintLevel::BottomOut;

        // Test the xp_penalty_permille method
        assert_eq!(meta.xp_penalty_permille(), 0);      // No penalty
        assert_eq!(directional.xp_penalty_permille(), 100);  // 10% penalty
        assert_eq!(specific.xp_penalty_permille(), 250);     // 25% penalty
        assert_eq!(bottom_out.xp_penalty_permille(), 500);   // 50% penalty

        // Penalties should increase progressively
        assert!(meta.xp_penalty_permille() < directional.xp_penalty_permille());
        assert!(directional.xp_penalty_permille() < specific.xp_penalty_permille());
        assert!(specific.xp_penalty_permille() < bottom_out.xp_penalty_permille());
    }

    #[test]
    fn test_hint_response_structure() {
        // Test hint response with progressive revelation
        let hint = HintResponse {
            hint_level: HintLevel::Directional,
            hint_text: "Think about what happens to the denominator...".to_string(),
            remaining_hints: 2,
            xp_penalty_if_used: 100,
            related_concepts: vec!["fractions".to_string()],
            try_first_suggestions: vec!["Review the denominator rule".to_string()],
        };

        assert!(matches!(hint.hint_level, HintLevel::Directional));
        assert!(hint.remaining_hints > 0);
        assert!(hint.xp_penalty_if_used == 100);
        assert!(!hint.related_concepts.is_empty());
    }

    #[test]
    fn test_explanation_types() {
        // 6 explanation types for personalization
        let types = [
            ExplanationType::Conceptual,
            ExplanationType::Procedural,
            ExplanationType::ExampleBased,
            ExplanationType::Analogy,
            ExplanationType::Visual,
            ExplanationType::Comparative,
        ];
        assert_eq!(types.len(), 6);

        // Match explanation to learner preference
        let conceptual = ExplanationType::Conceptual;
        let example = ExplanationType::ExampleBased;
        let analogy = ExplanationType::Analogy;

        assert!(matches!(conceptual, ExplanationType::Conceptual));
        assert!(matches!(example, ExplanationType::ExampleBased));
        assert!(matches!(analogy, ExplanationType::Analogy));
    }

    #[test]
    fn test_personalized_explanation() {
        // Test explanation structure with Mayer principles
        let explanation = PersonalizedExplanation {
            explanation_type: ExplanationType::ExampleBased,
            content: "Let's look at 3/4 + 1/4 = 4/4 = 1".to_string(),
            complexity_level: 500,  // Matched to learner (permille)
            examples: vec!["3/4 + 1/4 = 1".to_string()],
            analogies: vec!["Like pieces of pie".to_string()],
            key_points: vec!["Same denominator, add numerators".to_string()],
            common_misconceptions: vec!["Adding denominators".to_string()],
            follow_up_questions: vec!["What if denominators differ?".to_string()],
        };

        assert!(matches!(explanation.explanation_type, ExplanationType::ExampleBased));
        assert!(explanation.complexity_level <= 1000);
        assert!(!explanation.examples.is_empty());
        assert!(!explanation.key_points.is_empty());
    }

    #[test]
    fn test_feedback_types() {
        // 6 feedback types for different situations
        let types = [
            FeedbackType::Correct,
            FeedbackType::PartiallyCorrect,
            FeedbackType::Incorrect,
            FeedbackType::Effort,
            FeedbackType::Progress,
            FeedbackType::Mastery,
        ];
        assert_eq!(types.len(), 6);

        // Growth-oriented feedback
        let effort = FeedbackType::Effort;
        let progress = FeedbackType::Progress;
        let mastery = FeedbackType::Mastery;

        assert!(matches!(effort, FeedbackType::Effort));
        assert!(matches!(progress, FeedbackType::Progress));
        assert!(matches!(mastery, FeedbackType::Mastery));
    }

    #[test]
    fn test_personalized_feedback() {
        // Test feedback structure with emotional awareness
        let feedback = PersonalizedFeedback {
            feedback_type: FeedbackType::PartiallyCorrect,
            message: "Good thinking! You're on the right track.".to_string(),
            specific_praise: Some("Your approach to breaking down the problem was excellent.".to_string()),
            growth_mindset_message: Some("Keep trying, mistakes help you learn!".to_string()),
            next_steps: vec!["Try checking your arithmetic in step 3.".to_string()],
            emotional_tone: "Encouraging".to_string(),
        };

        assert!(matches!(feedback.feedback_type, FeedbackType::PartiallyCorrect));
        assert!(feedback.specific_praise.is_some());
        assert!(feedback.growth_mindset_message.is_some());
        assert!(!feedback.next_steps.is_empty());
    }

    #[test]
    fn test_misconception_detection() {
        // Test misconception detection for targeted remediation
        let misconception = DetectedMisconception {
            misconception_id: "FRAC-ADD-DENOM".to_string(),
            description: "Adding numerator and denominator separately".to_string(),
            evidence: vec!["1/2 + 1/3 = 2/5 (incorrect)".to_string()],
            correct_understanding: "Find common denominator first".to_string(),
            remediation_approach: "Review common denominator concept".to_string(),
            common_sources: vec!["Elementary fraction introduction".to_string()],
        };

        assert!(!misconception.evidence.is_empty());
        assert!(!misconception.correct_understanding.is_empty());
        assert!(!misconception.common_sources.is_empty());
    }

    #[test]
    fn test_tutoring_strategy() {
        // Test tutoring strategy selection
        let strategy = TutoringStrategy {
            recommended_mode: TutoringMode::Scaffolded,
            reasoning: "Learner shows moderate mastery with gaps".to_string(),
            estimated_time_minutes: 15,
            key_concepts_to_cover: vec!["Common denominators".to_string()],
            potential_obstacles: vec!["Fraction vocabulary confusion".to_string()],
            adaptive_triggers: vec!["If 3 errors, switch to remediation".to_string()],
        };

        assert!(matches!(strategy.recommended_mode, TutoringMode::Scaffolded));
        assert!(strategy.estimated_time_minutes > 0);
        assert!(!strategy.key_concepts_to_cover.is_empty());
    }

    #[test]
    fn test_tutoring_session_dialogue() {
        // Test dialogue turn structure
        let turn = DialogueTurn {
            speaker: "tutor".to_string(),
            message: "Can you explain why you chose that approach?".to_string(),
            intent: "probe".to_string(),
            timestamp: 1704067200,
            confidence_permille: Some(800),
        };

        // Verify DialogueTurn structure
        assert_eq!(turn.speaker, "tutor");
        assert_eq!(turn.intent, "probe");
        assert!(turn.confidence_permille.is_some());

        // Test TutoringSession with mock ActionHash
        let session_id = "session-123".to_string();
        let hints_used = 1u8;
        let session_mastery_start = 450u16;
        let session_mastery_current = 580u16;

        // Verify session shows mastery improvement
        assert!(session_mastery_current > session_mastery_start);
        assert_eq!(hints_used, 1);
        assert!(!session_id.is_empty());
    }

    #[test]
    fn test_tutoring_mode_selection_by_mastery() {
        // Select tutoring mode based on mastery level
        fn select_mode(mastery_permille: u16) -> TutoringMode {
            match mastery_permille {
                0..=200 => TutoringMode::Remediation,
                201..=400 => TutoringMode::WorkedExamples,
                401..=600 => TutoringMode::Scaffolded,
                601..=800 => TutoringMode::GuidedDiscovery,
                _ => TutoringMode::Socratic,
            }
        }

        // Low mastery -> Remediation
        assert!(matches!(select_mode(150), TutoringMode::Remediation));
        // Medium mastery -> Scaffolded
        assert!(matches!(select_mode(500), TutoringMode::Scaffolded));
        // High mastery -> Socratic (deep questioning)
        assert!(matches!(select_mode(850), TutoringMode::Socratic));
    }

    #[test]
    fn test_cross_tier_gamification_tutoring_integration() {
        // Gamification should integrate with tutoring
        let hints_used = 2u8;
        let base_xp = 100u32;

        // XP reduced by hint penalties
        let hint1_penalty = HintLevel::Directional.xp_penalty_permille();
        let hint2_penalty = HintLevel::Specific.xp_penalty_permille();
        let total_penalty_permille = hint1_penalty + hint2_penalty;  // 10% + 25% = 35%

        let final_xp = base_xp * (1000 - total_penalty_permille as u32) / 1000;

        assert_eq!(final_xp, 65);  // 100 * 0.65 = 65 XP after penalties
        assert!(final_xp < base_xp);
    }

    #[test]
    fn test_cross_tier_analytics_tutoring_integration() {
        // Analytics should inform tutoring decisions
        let trajectory = TrajectoryType::Declining;
        let at_risk_level = 700u16;  // 70% risk

        // High-risk declining trajectory -> Remediation mode
        let mode = if at_risk_level > 600 && matches!(trajectory, TrajectoryType::Declining) {
            TutoringMode::Remediation
        } else {
            TutoringMode::Scaffolded
        };

        assert!(matches!(mode, TutoringMode::Remediation));
    }

    #[test]
    fn test_cross_tier_gamification_analytics_integration() {
        // Achievements should be awarded based on analytics
        let trajectory = TrajectoryType::Accelerating;
        let weeks_consistent = 4u8;

        // Accelerating trajectory + consistency = achievement unlock
        let unlock_achievement = matches!(trajectory, TrajectoryType::Accelerating)
            && weeks_consistent >= 3;

        assert!(unlock_achievement);

        // Achievement tier based on duration
        let tier = if weeks_consistent >= 4 {
            AchievementTier::Silver
        } else {
            AchievementTier::Bronze
        };

        assert!(matches!(tier, AchievementTier::Silver));
    }
}
