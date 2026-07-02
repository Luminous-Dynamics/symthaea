// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learning science functions for the adaptive learning coordinator zome.
//!
//! Contains 29 learning science framework implementations:
//! Bloom's Taxonomy, Transfer of Learning, Elaboration, Worked Examples,
//! Expertise Reversal, Desirable Difficulties, Dual Coding, Retrieval Practice,
//! Hypercorrection, Pre-testing, Productive Failure, SDT, Mindset,
//! Attention, Critical Thinking, Source Evaluation, Bias Detection,
//! Socratic Dialogue, Metacognition, Collaboration, Creativity,
//! Inquiry, Emotional State, Deliberate Practice, SEL,
//! Stealth Assessment, Dynamic Assessment, UDL.
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;
use crate::helpers::truncate_text;
use crate::CalibrationTrend;

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
pub(crate) fn check_cognitive_level_unlock(input: SkillMasteryByLevel) -> ExternResult<Option<CognitiveLevel>> {
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
pub(crate) fn calculate_composite_mastery(input: SkillMasteryByLevel) -> ExternResult<u16> {
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
pub(crate) fn calculate_transfer_readiness(input: TransferMap) -> ExternResult<u16> {
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
pub(crate) fn check_deep_mastery(input: TransferMap) -> ExternResult<bool> {
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
pub(crate) fn generate_elaboration_prompt(input: GenerateElaborationInput) -> ExternResult<ElaborationPrompt> {
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
pub(crate) fn assess_elaboration_quality(quality_score: u16) -> ExternResult<ElaborationResult> {
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
pub(crate) fn calculate_worked_example_ratio(mastery_permille: u16) -> u16 {
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
pub(crate) fn recommend_content_format(input: WorkedExampleInput) -> ExternResult<WorkedExampleRecommendation> {
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
pub(crate) fn detect_expertise_reversal(input: ExpertiseCheckInput) -> ExternResult<ExpertiseReversalResult> {
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
pub(crate) fn calculate_desirable_difficulties(input: DesirableDifficultyInput) -> ExternResult<DesirableDifficultyProfile> {
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
pub(crate) fn recommend_dual_coding(input: DualCodingInput) -> ExternResult<DualCodingRecommendation> {
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
pub(crate) fn calculate_dual_coding_load(modalities: Vec<ContentModality>) -> ExternResult<u16> {
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
pub(crate) fn calculate_retrieval_schedule(input: RetrievalPracticeInput) -> ExternResult<RetrievalPracticeResult> {
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
pub(crate) fn analyze_hypercorrection(input: HypercorrectionInput) -> ExternResult<HypercorrectionAnalysis> {
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
pub(crate) fn analyze_pre_test_benefit(input: PreTestInput) -> ExternResult<PreTestAnalysis> {
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
pub(crate) fn analyze_productive_failure(input: ProductiveFailureInput) -> ExternResult<ProductiveFailureAnalysis> {
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
pub(crate) fn assess_sdt_needs(input: SDTInput) -> ExternResult<SDTAssessment> {
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
pub(crate) fn assess_mindset(input: MindsetInput) -> ExternResult<MindsetAssessment> {
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
pub(crate) fn assess_attention(input: AttentionInput) -> ExternResult<AttentionAssessment> {
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
pub(crate) fn analyze_argument(input: ArgumentInput) -> ExternResult<ArgumentAnalysis> {
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
pub(crate) fn evaluate_source(input: SourceEvaluationInput) -> ExternResult<SourceCredibility> {
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
pub(crate) fn detect_biases(input: BiasDetectionInput) -> ExternResult<BiasAnalysis> {
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
pub(crate) fn generate_socratic_questions(input: SocraticInput) -> ExternResult<Vec<SocraticDialogue>> {
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

// truncate_text is imported from crate::helpers

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
pub(crate) fn assess_metacognition(input: MetacognitiveInput) -> ExternResult<MetacognitiveAssessment> {
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
pub(crate) fn assess_collaboration(input: CollaborativeInput) -> ExternResult<CollaborativeAssessment> {
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
pub(crate) fn assess_creativity(input: CreativityInput) -> ExternResult<CreativityAssessment> {
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
pub(crate) fn assess_inquiry(input: InquiryInput) -> ExternResult<InquiryAssessment> {
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
pub(crate) fn assess_emotional_state(input: EmotionAssessmentInput) -> ExternResult<EmotionAssessment> {
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
pub(crate) fn assess_deliberate_practice(input: DeliberatePracticeInput) -> ExternResult<DeliberatePracticeAssessment> {
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
pub(crate) fn assess_sel_competencies(input: SELInput) -> ExternResult<SELAssessment> {
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
pub(crate) fn perform_stealth_assessment(input: StealthAssessmentInput) -> ExternResult<StealthAssessment> {
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
pub(crate) fn perform_dynamic_assessment(input: DynamicAssessmentInput) -> ExternResult<DynamicAssessmentResult> {
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
pub(crate) fn assess_udl_needs(input: UDLInput) -> ExternResult<UDLAssessment> {
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
