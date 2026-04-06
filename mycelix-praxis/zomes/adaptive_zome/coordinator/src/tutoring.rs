// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intelligent tutoring system functions for the adaptive learning coordinator zome.
//!
//! Contains: hint generation, explanation generation, personalized feedback,
//! misconception detection, and tutoring strategy recommendation.
//!
//! Research: VanLehn (2011), Graesser (2004) - Intelligent Tutoring Systems
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;

// ============== Types ==============

/// Tutoring mode
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TutoringMode {
    /// Socratic questioning
    Socratic,
    /// Direct instruction
    Direct,
    /// Guided discovery
    GuidedDiscovery,
    /// Worked examples
    WorkedExamples,
    /// Scaffolded practice
    Scaffolded,
    /// Error-focused remediation
    Remediation,
}

/// Hint level (progressive revelation)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum HintLevel {
    /// Metacognitive prompt ("Think about...")
    Metacognitive,
    /// Direction hint ("Consider the relationship...")
    Directional,
    /// Specific hint ("The formula involves...")
    Specific,
    /// Bottom-out hint (explicit answer with explanation)
    BottomOut,
}

impl HintLevel {
    pub fn xp_penalty_permille(&self) -> u16 {
        match self {
            HintLevel::Metacognitive => 0,
            HintLevel::Directional => 100,
            HintLevel::Specific => 250,
            HintLevel::BottomOut => 500,
        }
    }
}

/// Hint request and response
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintResponse {
    pub hint_level: HintLevel,
    pub hint_text: String,
    pub remaining_hints: u8,
    pub xp_penalty_if_used: u16,
    pub related_concepts: Vec<String>,
    pub try_first_suggestions: Vec<String>,
}

/// Explanation type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExplanationType {
    /// Conceptual explanation (what and why)
    Conceptual,
    /// Procedural explanation (how)
    Procedural,
    /// Example-based explanation
    ExampleBased,
    /// Analogy-based explanation
    Analogy,
    /// Visual/diagrammatic explanation
    Visual,
    /// Comparative explanation (vs other concepts)
    Comparative,
}

/// Personalized explanation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalizedExplanation {
    pub explanation_type: ExplanationType,
    pub content: String,
    pub complexity_level: u16, // Matched to learner
    pub examples: Vec<String>,
    pub analogies: Vec<String>,
    pub key_points: Vec<String>,
    pub common_misconceptions: Vec<String>,
    pub follow_up_questions: Vec<String>,
}

/// Feedback type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedbackType {
    /// Correct answer acknowledgment
    Correct,
    /// Partially correct with guidance
    PartiallyCorrect,
    /// Incorrect with explanation
    Incorrect,
    /// Encouragement for effort
    Effort,
    /// Progress celebration
    Progress,
    /// Mastery acknowledgment
    Mastery,
}

/// Personalized feedback
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersonalizedFeedback {
    pub feedback_type: FeedbackType,
    pub message: String,
    pub specific_praise: Option<String>,
    pub growth_mindset_message: Option<String>,
    pub next_steps: Vec<String>,
    pub emotional_tone: String, // "encouraging", "celebratory", "supportive", etc.
}

/// Tutoring dialogue turn
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DialogueTurn {
    pub speaker: String, // "tutor" or "learner"
    pub message: String,
    pub intent: String, // "question", "answer", "hint", "explanation", etc.
    pub timestamp: i64,
    pub confidence_permille: Option<u16>,
}

/// Tutoring session state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TutoringSession {
    pub session_id: String,
    pub skill_hash: ActionHash,
    pub tutoring_mode: TutoringMode,
    pub dialogue_history: Vec<DialogueTurn>,
    pub hints_used: u8,
    pub explanations_requested: u8,
    pub current_problem_attempts: u8,
    pub session_mastery_start: u16,
    pub session_mastery_current: u16,
    pub emotional_state_detected: String,
}

/// Misconception detected
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectedMisconception {
    pub misconception_id: String,
    pub description: String,
    pub evidence: Vec<String>,
    pub correct_understanding: String,
    pub remediation_approach: String,
    pub common_sources: Vec<String>,
}

/// Input for hint request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintRequestInput {
    pub skill_hash: ActionHash,
    pub problem_id: String,
    pub current_attempt: String,
    pub hints_already_used: u8,
    pub time_stuck_seconds: u32,
}

/// Input for explanation request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplanationRequestInput {
    pub skill_hash: ActionHash,
    pub concept: String,
    pub learner_mastery: u16,
    pub preferred_explanation_type: Option<ExplanationType>,
    pub vark_style: LearningStyle,
    pub previous_explanations_seen: u8,
}

/// Input for feedback generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeedbackInput {
    pub response_correct: bool,
    pub partial_credit_permille: Option<u16>,
    pub learner_confidence: u16,
    pub attempt_number: u8,
    pub time_spent_seconds: u32,
    pub skill_mastery: u16,
    pub recent_streak: i8, // Positive = correct streak, negative = error streak
}

/// Input for misconception detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MisconceptionInput {
    pub skill_hash: ActionHash,
    pub error_pattern: Vec<String>,
    pub incorrect_answers: Vec<String>,
    pub time_pattern: Vec<u32>,
}

/// Tutoring strategy recommendation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TutoringStrategy {
    pub recommended_mode: TutoringMode,
    pub reasoning: String,
    pub estimated_time_minutes: u16,
    pub key_concepts_to_cover: Vec<String>,
    pub potential_obstacles: Vec<String>,
    pub adaptive_triggers: Vec<String>,
}

/// Input for tutoring strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TutoringStrategyInput {
    pub skill_hash: ActionHash,
    pub learner_mastery: u16,
    pub learning_style: LearningStyle,
    pub recent_errors: u8,
    pub time_available_minutes: u16,
    pub goal_type: GoalType,
}

// ============== Functions ==============

/// Request a hint (progressive revelation)
pub(crate) fn request_hint(input: HintRequestInput) -> ExternResult<HintResponse> {
    // Progressive hint revelation
    let hint_level = match input.hints_already_used {
        0 => HintLevel::Metacognitive,
        1 => HintLevel::Directional,
        2 => HintLevel::Specific,
        _ => HintLevel::BottomOut,
    };

    let (hint_text, try_first) = match hint_level {
        HintLevel::Metacognitive => (
            "Think about what you already know about this concept. What patterns do you see?".to_string(),
            vec!["Review the related concept".to_string(), "Try breaking the problem into smaller parts".to_string()],
        ),
        HintLevel::Directional => (
            "Consider the relationship between the input and output. What transformation is happening?".to_string(),
            vec!["Look at similar examples you've solved".to_string()],
        ),
        HintLevel::Specific => (
            "The key insight is that this problem requires applying the principle you learned in the previous section.".to_string(),
            vec!["Focus on applying the formula step by step".to_string()],
        ),
        HintLevel::BottomOut => (
            "Here's the approach: First identify the pattern, then apply the formula, finally check your answer.".to_string(),
            vec![],
        ),
    };

    Ok(HintResponse {
        hint_level: hint_level.clone(),
        hint_text,
        remaining_hints: 3u8.saturating_sub(input.hints_already_used),
        xp_penalty_if_used: hint_level.xp_penalty_permille(),
        related_concepts: vec!["Pattern Recognition".to_string(), "Problem Decomposition".to_string()],
        try_first_suggestions: try_first,
    })
}

/// Request personalized explanation
pub(crate) fn request_explanation(input: ExplanationRequestInput) -> ExternResult<PersonalizedExplanation> {
    // Select explanation type based on preferences and learning style
    let explanation_type = input.preferred_explanation_type.unwrap_or_else(|| {
        match input.vark_style {
            LearningStyle::Visual => ExplanationType::Visual,
            LearningStyle::Auditory => ExplanationType::Analogy,
            LearningStyle::ReadingWriting => ExplanationType::Conceptual,
            LearningStyle::Kinesthetic => ExplanationType::ExampleBased,
            LearningStyle::Multimodal => ExplanationType::Comparative,
        }
    });

    // Adjust complexity based on mastery
    let complexity = input.learner_mastery;

    Ok(PersonalizedExplanation {
        explanation_type,
        content: format!("Understanding '{}' at your level...", input.concept),
        complexity_level: complexity,
        examples: vec![
            "Example 1: Simple case".to_string(),
            "Example 2: With variation".to_string(),
        ],
        analogies: vec![
            "Think of it like...".to_string(),
        ],
        key_points: vec![
            "Key point 1: The fundamental principle".to_string(),
            "Key point 2: Common application".to_string(),
            "Key point 3: Edge cases to watch".to_string(),
        ],
        common_misconceptions: vec![
            "Many learners initially think X, but actually Y".to_string(),
        ],
        follow_up_questions: vec![
            "Can you think of another example?".to_string(),
            "How would this apply to...?".to_string(),
        ],
    })
}

/// Generate personalized feedback
pub(crate) fn generate_feedback(input: FeedbackInput) -> ExternResult<PersonalizedFeedback> {
    let (feedback_type, message, praise, growth_msg) = if input.response_correct {
        if input.attempt_number == 1 && input.time_spent_seconds < 30 {
            (
                FeedbackType::Correct,
                "Excellent! You solved that quickly and correctly.".to_string(),
                Some("Great problem-solving speed!".to_string()),
                None,
            )
        } else if input.attempt_number > 2 {
            (
                FeedbackType::Correct,
                "You got it! Persistence paid off.".to_string(),
                Some("Great job not giving up!".to_string()),
                Some("Struggling and succeeding builds strong neural pathways.".to_string()),
            )
        } else {
            (
                FeedbackType::Correct,
                "Correct! Well done.".to_string(),
                None,
                None,
            )
        }
    } else if let Some(partial) = input.partial_credit_permille {
        if partial > 700 {
            (
                FeedbackType::PartiallyCorrect,
                "Very close! You have the right approach, just a small adjustment needed.".to_string(),
                Some("Your reasoning is solid.".to_string()),
                None,
            )
        } else {
            (
                FeedbackType::PartiallyCorrect,
                "You're on the right track. Let's build on what you got right.".to_string(),
                None,
                Some("Each attempt teaches you something valuable.".to_string()),
            )
        }
    } else {
        // Handle overconfidence with care
        if input.learner_confidence > 800 {
            (
                FeedbackType::Incorrect,
                "Not quite. Let's examine this together - there might be a misconception to address.".to_string(),
                None,
                Some("High-confidence errors are great learning opportunities - they stick better when corrected!".to_string()),
            )
        } else {
            (
                FeedbackType::Incorrect,
                "That's not it, but don't worry. Let's work through this step by step.".to_string(),
                Some("Your effort is what counts.".to_string()),
                Some("Mistakes are how we learn. Each one brings you closer to understanding.".to_string()),
            )
        }
    };

    let next_steps = if input.response_correct {
        vec!["Try the next problem".to_string(), "Increase difficulty".to_string()]
    } else {
        vec!["Review the concept".to_string(), "Try a simpler version".to_string(), "Request a hint".to_string()]
    };

    let emotional_tone = if input.response_correct { "celebratory" } else { "supportive" };

    Ok(PersonalizedFeedback {
        feedback_type,
        message,
        specific_praise: praise,
        growth_mindset_message: growth_msg,
        next_steps,
        emotional_tone: emotional_tone.to_string(),
    })
}

/// Detect misconceptions from error patterns
pub(crate) fn detect_misconceptions(input: MisconceptionInput) -> ExternResult<Vec<DetectedMisconception>> {
    let mut misconceptions = Vec::new();

    // Analyze error patterns to detect common misconceptions
    if input.error_pattern.len() > 2 {
        misconceptions.push(DetectedMisconception {
            misconception_id: "common_1".to_string(),
            description: "Appears to be applying rule incorrectly in edge cases".to_string(),
            evidence: input.error_pattern.iter().take(3).cloned().collect(),
            correct_understanding: "The rule applies differently when conditions change".to_string(),
            remediation_approach: "Provide explicit boundary conditions with examples".to_string(),
            common_sources: vec![
                "Overgeneralizing from simple cases".to_string(),
                "Missing prerequisite concept".to_string(),
            ],
        });
    }

    Ok(misconceptions)
}

/// Get recommended tutoring strategy
pub(crate) fn get_tutoring_strategy(input: TutoringStrategyInput) -> ExternResult<TutoringStrategy> {
    // Select mode based on learner state
    let (mode, reasoning) = if input.recent_errors > 3 {
        (TutoringMode::Remediation, "Recent errors indicate need for targeted remediation")
    } else if input.learner_mastery < 300 {
        (TutoringMode::WorkedExamples, "Low mastery - start with worked examples to build foundation")
    } else if input.learner_mastery < 600 {
        (TutoringMode::Scaffolded, "Building mastery - scaffolded practice with gradual release")
    } else if matches!(input.goal_type, GoalType::CourseCompletion) {
        (TutoringMode::GuidedDiscovery, "Ready for discovery-based learning")
    } else {
        (TutoringMode::Socratic, "High mastery - Socratic method to deepen understanding")
    };

    Ok(TutoringStrategy {
        recommended_mode: mode,
        reasoning: reasoning.to_string(),
        estimated_time_minutes: input.time_available_minutes.min(45),
        key_concepts_to_cover: vec!["Core concept".to_string(), "Application".to_string()],
        potential_obstacles: vec!["Common misconception about edge cases".to_string()],
        adaptive_triggers: vec![
            "If 3+ errors in a row, switch to worked examples".to_string(),
            "If quick success, increase difficulty".to_string(),
        ],
    })
}
