// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cognitive adaptivity engine for EduNet.
//!
//! This module is the core intelligence that makes learning content morph in
//! real-time based on the student's cognitive state. It consumes signals from
//! the Symthaea consciousness engine (Phi, free energy, neuromodulators, FEP
//! motor commands) and produces concrete adaptation decisions:
//!
//! 1. **What** to present (content selection, difficulty adjustment)
//! 2. **How** to present it (modality, text complexity, vocabulary level)
//! 3. **When** to intervene (frustration detection, grounding, breaks)
//! 4. **Who** to pair (peer matching based on cognitive compatibility)
//!
//! # Signal derivation
//!
//! From the three Spore neuromodulators (DA, 5-HT, NE) we derive:
//! - **Cortisol proxy**: NE * (1 - 5HT)  -- stress
//! - **Acetylcholine proxy**: consciousness_level  -- focused attention
//! - **Oxytocin proxy**: 5HT * 0.6 + DA * 0.4  -- social bonding readiness
//! - **Valence**: DA - cortisol  -- frustrated (-1) to happy (+1)
//! - **Arousal**: NE + DA * 0.5  -- calm (0) to excited (1)
//! - **Markov permeability**: sigmoid(5HT + oxytocin - NE - cortisol)  -- openness to input
//!
//! # Frustration model
//!
//! Frustration is detected as the product of negative valence, high arousal,
//! and consecutive failures. This maps to the Yerkes-Dodson inverted-U: when
//! arousal is high but performance is failing, the student has crossed the
//! optimal zone. The engine intervenes with grounding exercises, reappraisal,
//! or scaffolding depending on severity and age.
//!
//! # ZPD and Markov permeability
//!
//! Vygotsky's Zone of Proximal Development is dynamically modulated by the
//! Markov blanket permeability signal. High permeability (open blanket) means
//! the student is receptive to new information -- wider ZPD. Low permeability
//! (closed/fatigued) means review-only -- narrow ZPD. The `difficulty_delta`
//! is scaled by permeability so that fatigued students get gentler adjustments.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Real-time cognitive state derived from consciousness + neuromodulators.
///
/// Built from `ConsciousnessState` signals each tick by the adaptivity layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveState {
    /// Phi (integrated information) -- consciousness level.
    pub phi: f32,
    /// Free energy -- surprise/prediction error. High = confused.
    pub free_energy: f32,
    /// Valence: DA - cortisol. Range: roughly -1 to 1 (frustrated to happy).
    pub valence: f32,
    /// Arousal: NE + DA * 0.5. Range: 0 to ~1.5, clamped to 0-1.
    pub arousal: f32,
    /// Acetylcholine proxy (= consciousness_level). Focused attention.
    pub focus: f32,
    /// Dopamine. Motivation / reward prediction.
    pub motivation: f32,
    /// Cortisol proxy: NE * (1 - 5HT). Stress level.
    pub stress: f32,
    /// Oxytocin proxy: 5HT * 0.6 + DA * 0.4. Social readiness.
    pub social_readiness: f32,
    /// Markov blanket permeability: sigmoid(5HT + oxytocin - NE - cortisol).
    /// How open the student is to new information.
    pub permeability: f32,
    /// Latest FEP motor command from the consciousness engine.
    pub motor_command: MotorCommand,
}

/// FEP motor command mapped to educational actions.
///
/// The Symthaea Spore engine emits 8 motor command types from its active
/// inference loop. Each maps to a pedagogical strategy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MotorCommand {
    /// Switch topic or modality -- attention has wandered.
    AttentionShift,
    /// Adjust learning rate -- slow down or speed up.
    LearningRateAdjust,
    /// Present novel/surprising content -- seek information gain.
    ExplorationTrigger,
    /// Metacognitive pause ("what did you learn?").
    ReflectionInitiate,
    /// Review / spaced repetition -- consolidate.
    MemoryConsolidate,
    /// Start fresh approach to problem -- prior model failed.
    ExpectationReset,
    /// Student action (answer, draw, type).
    MotorOutput,
    /// Continue current activity.
    NoOp,
}

// ---------------------------------------------------------------------------
// Adaptation output types
// ---------------------------------------------------------------------------

/// Content adaptation decisions produced by the engine.
///
/// Sent to the rendering layer and logged for the teacher dashboard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentAdaptation {
    /// How to modify text complexity.
    pub text_complexity: TextComplexity,
    /// What presentation modality to use.
    pub modality: Modality,
    /// Difficulty adjustment from baseline. Range: -1.0 to +1.0.
    pub difficulty_delta: f32,
    /// Whether to trigger an intervention.
    pub intervention: Option<Intervention>,
    /// Peer pairing suggestion, if appropriate.
    pub peer_suggestion: Option<PeerSuggestion>,
    /// Human-readable reasoning trace (for teacher dashboard).
    pub reasoning: String,
}

/// Text complexity levels for content rewriting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TextComplexity {
    /// Full standard text as authored.
    Standard,
    /// Simplified vocabulary, shorter sentences.
    Simplified,
    /// Minimal text, heavy visual/pictorial support.
    Minimal,
    /// Rewritten to incorporate student interests.
    Personalized { interest_topic: String },
}

/// Presentation modality.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Modality {
    /// Reading text.
    Text,
    /// Diagrams, arrays, pictures.
    Visual,
    /// Read-aloud / audio.
    Auditory,
    /// Drag-and-drop, manipulatives, hands-on.
    Kinesthetic,
    /// Multiple modalities combined.
    MultiModal,
}

/// Intervention types triggered by the adaptivity engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Intervention {
    /// Pause learning, do a grounding exercise.
    Grounding {
        duration_seconds: u32,
        exercise: String,
    },
    /// Cognitive reappraisal -- reframe the difficulty.
    Reappraisal { message: String },
    /// Fall back to easier review material.
    Scaffolding { fallback_skill: String },
    /// Celebrate progress and push harder.
    Challenge { message: String },
    /// Mandatory break.
    Break {
        duration_minutes: u32,
        reason: String,
    },
    /// Suggest collaborative peer work.
    PeerWork { reason: String },
}

/// Suggestion for peer pairing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerSuggestion {
    /// Why peer work is recommended.
    pub reason: String,
    /// Criteria for selecting a compatible peer.
    pub peer_criteria: PeerCriteria,
    /// Suggested collaboration duration.
    pub duration_minutes: u32,
    /// Expected pedagogical benefit.
    pub expected_benefit: String,
}

/// Criteria for selecting a compatible peer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerCriteria {
    /// The skill the peer should have high mastery in.
    pub needs_high_mastery_in: String,
    /// Whether the peer should be patient (high Phi + low arousal).
    pub needs_patience: bool,
    /// Whether affect compatibility is required (not both frustrated).
    pub compatible_affect: bool,
}

// ---------------------------------------------------------------------------
// Signal derivation
// ---------------------------------------------------------------------------

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Build a `CognitiveState` from raw consciousness signals.
///
/// This is the bridge between `ConsciousnessState` (from the consciousness
/// module) and the adaptivity engine. Pure function, no Leptos dependency.
pub fn derive_cognitive_state(
    phi: f32,
    free_energy: f32,
    dopamine: f32,
    serotonin: f32,
    norepinephrine: f32,
    motor_command: MotorCommand,
) -> CognitiveState {
    let cortisol = (norepinephrine * (1.0 - serotonin)).clamp(0.0, 1.0);
    let acetylcholine = phi.clamp(0.0, 1.0);
    let oxytocin = (serotonin * 0.6 + dopamine * 0.4).clamp(0.0, 1.0);
    let valence = (dopamine - cortisol).clamp(-1.0, 1.0);
    let arousal = (norepinephrine + dopamine * 0.5).clamp(0.0, 1.0);
    let permeability = sigmoid(serotonin + oxytocin - norepinephrine - cortisol);

    CognitiveState {
        phi,
        free_energy,
        valence,
        arousal,
        focus: acetylcholine,
        motivation: dopamine,
        stress: cortisol,
        social_readiness: oxytocin,
        permeability,
        motor_command,
    }
}

// ---------------------------------------------------------------------------
// Core decision engine
// ---------------------------------------------------------------------------

/// Compute content adaptation from cognitive state and learning context.
///
/// This is the main entry point for the adaptivity engine. It takes the
/// student's real-time cognitive state, their current learning context, and
/// returns concrete adaptation decisions.
///
/// # Arguments
///
/// * `state` - Real-time cognitive state derived from consciousness signals.
/// * `current_skill` - Name of the skill being practiced (e.g. "multiplication").
/// * `current_mastery_permille` - Current mastery level (0-1000).
/// * `recent_accuracy` - Accuracy over the last 5 attempts (0.0-1.0).
/// * `consecutive_failures` - Number of consecutive wrong answers.
/// * `grade_ordinal` - Grade level (0=PreK, 1=K, 2=Grade1, ..., 13=Grade12).
/// * `student_interests` - List of interests for personalization.
pub fn adapt_content(
    state: &CognitiveState,
    current_skill: &str,
    current_mastery_permille: u16,
    recent_accuracy: f32,
    consecutive_failures: u32,
    grade_ordinal: u8,
    student_interests: &[String],
) -> ContentAdaptation {
    // === FRUSTRATION DETECTION ===
    // Valence dropping + arousal rising + consecutive failures = frustration.
    // Maps to the Yerkes-Dodson inverted-U: past the optimal arousal point.
    let frustration_score = (-state.valence).max(0.0)
        * state.arousal
        * (consecutive_failures as f32 / 3.0).min(1.0);

    if frustration_score > 0.7 {
        // HIGH FRUSTRATION: Intervene immediately.
        return ContentAdaptation {
            text_complexity: TextComplexity::Simplified,
            modality: Modality::Visual, // reduce cognitive load
            difficulty_delta: -0.5,
            intervention: Some(if state.stress > 0.65 {
                Intervention::Grounding {
                    duration_seconds: 60,
                    exercise: "Take 5 deep breaths. Breathe in for 4 counts, \
                               out for 4 counts."
                        .into(),
                }
            } else {
                Intervention::Reappraisal {
                    message: format!(
                        "This is tricky! {} is a skill that takes practice. \
                         Let's try a different way to look at it.",
                        current_skill
                    ),
                }
            }),
            peer_suggestion: None,
            reasoning: format!(
                "Frustration detected (score {:.2}). Reducing difficulty \
                 and providing emotional support.",
                frustration_score
            ),
        };
    }

    // === TEXT COMPLEXITY ===
    // Repeated failures + low accuracy → simplify or personalize.
    // Low focus → minimal text to reduce cognitive load.
    let text_complexity = if consecutive_failures >= 2 && recent_accuracy < 0.4 {
        if !student_interests.is_empty() {
            TextComplexity::Personalized {
                interest_topic: student_interests[0].clone(),
            }
        } else {
            TextComplexity::Simplified
        }
    } else if state.focus < 0.3 {
        TextComplexity::Minimal // low focus -> reduce text load
    } else {
        TextComplexity::Standard
    };

    // === MODALITY FROM COGNITIVE STATE ===
    let modality = if state.focus > 0.7 {
        Modality::Text // high focus -> can handle dense text
    } else if state.social_readiness > 0.6 {
        Modality::Kinesthetic // social readiness -> hands-on
    } else if state.arousal < 0.3 {
        Modality::Visual // low arousal -> engage visually
    } else {
        Modality::MultiModal
    };

    // === DIFFICULTY FROM FEP MOTOR COMMAND ===
    let difficulty_delta = match state.motor_command {
        MotorCommand::ExplorationTrigger => 0.3,  // seek novelty -> harder
        MotorCommand::LearningRateAdjust => -0.2, // slow down -> easier
        MotorCommand::ReflectionInitiate => 0.0,   // pause to think
        MotorCommand::MemoryConsolidate => -0.3,  // review mode -> easier
        _ => {
            // Default: use free energy as difficulty guide.
            // High free energy = lots of surprise = too hard.
            if state.free_energy > 0.7 {
                -0.3
            } else if state.free_energy < 0.3 {
                0.2 // too easy -> increase
            } else {
                0.0
            }
        }
    };

    // === MARKOV PERMEABILITY -> ZPD MAPPING ===
    // High permeability = open to new input = wider ZPD.
    // Low permeability = closed/fatigued = narrow ZPD (review only).
    let effective_difficulty = difficulty_delta * state.permeability;

    // === INTERVENTION CHECK ===
    let intervention = if state.stress > 0.65 {
        Some(Intervention::Break {
            duration_minutes: if grade_ordinal <= 1 { 10 } else { 5 },
            reason: "Your brain has been working hard. A short break helps \
                     you learn better!"
                .into(),
        })
    } else if state.motor_command == MotorCommand::ReflectionInitiate {
        Some(Intervention::Challenge {
            message: "Nice work! Can you explain how you solved that in \
                      your own words?"
                .into(),
        })
    } else if state.phi > 0.6 && current_mastery_permille > 850 {
        Some(Intervention::Challenge {
            message: "You're really getting this! Ready for something harder?"
                .into(),
        })
    } else {
        None
    };

    // === PEER SUGGESTION ===
    let peer_suggestion =
        if state.social_readiness > 0.6
            && consecutive_failures >= 3
            && frustration_score < 0.5 // not too frustrated for social work
        {
            Some(PeerSuggestion {
                reason: format!(
                    "{} might be easier with a study buddy",
                    current_skill
                ),
                peer_criteria: PeerCriteria {
                    needs_high_mastery_in: current_skill.to_string(),
                    needs_patience: true,
                    compatible_affect: true,
                },
                duration_minutes: 10,
                expected_benefit: "Students who explain concepts to peers \
                                   deepen their own understanding"
                    .into(),
            })
        } else {
            None
        };

    // === REASONING TRACE (for teacher dashboard) ===
    let complexity_label = match &text_complexity {
        TextComplexity::Standard => "standard",
        TextComplexity::Simplified => "simplified",
        TextComplexity::Minimal => "minimal",
        TextComplexity::Personalized { .. } => "personalized",
    };
    let modality_label = match &modality {
        Modality::Text => "text",
        Modality::Visual => "visual",
        Modality::Auditory => "auditory",
        Modality::Kinesthetic => "kinesthetic",
        Modality::MultiModal => "multi",
    };
    let sign = if effective_difficulty >= 0.0 { "+" } else { "" };

    let reasoning = format!(
        "Phi={:.2}, FE={:.2}, valence={:.2}, arousal={:.2}, stress={:.2}, \
         permeability={:.2}. Motor: {:?}. Mastery: {}\u{2030}. Accuracy: {:.0}%. \
         Failures: {}. -> {} complexity, {} modality, difficulty {}{:.1}",
        state.phi,
        state.free_energy,
        state.valence,
        state.arousal,
        state.stress,
        state.permeability,
        state.motor_command,
        current_mastery_permille,
        recent_accuracy * 100.0,
        consecutive_failures,
        complexity_label,
        modality_label,
        sign,
        effective_difficulty,
    );

    ContentAdaptation {
        text_complexity,
        modality,
        difficulty_delta: effective_difficulty,
        intervention,
        peer_suggestion,
        reasoning,
    }
}

// ---------------------------------------------------------------------------
// Word problem rewriter
// ---------------------------------------------------------------------------

/// Rewrite a word problem based on text complexity and student interests.
///
/// This is a rule-based rewriter for immediate use. When the Broca language
/// center (Symthaea) is available via Spore WASM, it can produce richer
/// contextual rewrites. This version handles the common cases.
pub fn rewrite_problem(
    original: &str,
    complexity: &TextComplexity,
    grade_ordinal: u8,
) -> String {
    match complexity {
        TextComplexity::Standard => original.to_string(),
        TextComplexity::Simplified => simplify_text(original, grade_ordinal),
        TextComplexity::Minimal => extract_equation(original),
        TextComplexity::Personalized { interest_topic } => {
            personalize_text(original, interest_topic)
        }
    }
}

/// Simplify vocabulary and sentence structure.
///
/// Replaces complex words with grade-appropriate alternatives.
fn simplify_text(text: &str, grade_ordinal: u8) -> String {
    let mut result = text.to_string();

    let replacements: &[(&str, &str)] = &[
        ("purchased", "bought"),
        ("distributed", "shared"),
        ("remaining", "left"),
        ("calculate", "find"),
        ("determine", "find"),
        ("equivalent", "equal"),
        ("approximately", "about"),
        (
            "rectangle",
            if grade_ordinal <= 4 {
                "shape"
            } else {
                "rectangle"
            },
        ),
        (
            "multiplication",
            if grade_ordinal <= 3 {
                "times"
            } else {
                "multiplication"
            },
        ),
    ];

    for (complex, simple) in replacements {
        result = result.replace(complex, simple);
    }

    result
}

/// Extract just the mathematical operation from a word problem.
///
/// Strips narrative text and retains only digits and operators. Falls back
/// to the first sentence if no equation pattern is found.
fn extract_equation(text: &str) -> String {
    if text.contains('\u{00d7}')
        || text.contains('*')
        || text.contains('+')
        || text.contains('-')
    {
        text.chars()
            .filter(|c| {
                c.is_ascii_digit()
                    || *c == '\u{00d7}'
                    || *c == '*'
                    || *c == '+'
                    || *c == '-'
                    || *c == '='
                    || *c == ' '
            })
            .collect::<String>()
            .trim()
            .to_string()
    } else {
        // Can't extract equation -- just take the first sentence.
        text.split('.').next().unwrap_or(text).to_string()
    }
}

/// Replace context subjects with the student's interests.
///
/// Swaps generic nouns (bags, apples) for interest-themed equivalents.
fn personalize_text(text: &str, interest: &str) -> String {
    let generic_subjects = [
        "bags",
        "boxes",
        "groups",
        "baskets",
        "containers",
    ];
    let generic_items = [
        "apples", "oranges", "items", "objects", "things", "stickers",
        "marbles",
    ];

    let (subject, item) = match interest.to_lowercase().as_str() {
        "dinosaurs" => ("nests", "dinosaur eggs"),
        "space" => ("rockets", "astronauts"),
        "animals" => ("pens", "puppies"),
        "sports" => ("teams", "players"),
        "minecraft" => ("chests", "diamonds"),
        "pokemon" => ("packs", "cards"),
        "cooking" => ("bowls", "cookies"),
        "music" => ("bands", "instruments"),
        _ => ("groups", interest),
    };

    let mut result = text.to_string();
    for s in &generic_subjects {
        result = result.replace(s, subject);
    }
    for i in &generic_items {
        result = result.replace(i, item);
    }

    result
}

// ---------------------------------------------------------------------------
// Grounding exercises
// ---------------------------------------------------------------------------

/// Generate an age-appropriate grounding exercise.
///
/// Young students (PreK-Grade3) get physical/sensory exercises.
/// Older students get cognitive reappraisal framing.
pub fn grounding_exercise(grade_ordinal: u8, stress_level: f32) -> Intervention {
    if grade_ordinal <= 4 {
        // Young students: physical/sensory grounding.
        if stress_level > 0.8 {
            Intervention::Grounding {
                duration_seconds: 90,
                exercise: "Let's do the 5-4-3-2-1 game! \
                    Name 5 things you can see, 4 things you can touch, \
                    3 things you can hear, 2 things you can smell, \
                    and 1 thing you can taste."
                    .into(),
            }
        } else {
            Intervention::Grounding {
                duration_seconds: 60,
                exercise: "Let's take 5 big breaths together! \
                    Breathe in like you're smelling a flower... \
                    Breathe out like you're blowing out birthday candles..."
                    .into(),
            }
        }
    } else {
        // Older students: cognitive reappraisal.
        Intervention::Reappraisal {
            message: "It's normal to find this challenging. \
                Every expert was once a beginner. \
                Let's try breaking this down into smaller steps."
                .into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a CognitiveState with sensible defaults
    // -----------------------------------------------------------------------

    fn default_state() -> CognitiveState {
        CognitiveState {
            phi: 0.5,
            free_energy: 0.5,
            valence: 0.2,
            arousal: 0.4,
            focus: 0.5,
            motivation: 0.5,
            stress: 0.2,
            social_readiness: 0.4,
            permeability: 0.6,
            motor_command: MotorCommand::NoOp,
        }
    }

    // -----------------------------------------------------------------------
    // Frustration detection
    // -----------------------------------------------------------------------

    #[test]
    fn frustration_high_triggers_intervention() {
        // Negative valence + high arousal + consecutive failures = frustration.
        let state = CognitiveState {
            valence: -0.8,
            arousal: 0.9,
            stress: 0.3, // below grounding threshold
            ..default_state()
        };
        let result = adapt_content(&state, "fractions", 400, 0.1, 5, 5, &[]);
        assert!(result.intervention.is_some());
        // Should get reappraisal (stress < 0.65)
        assert!(matches!(
            result.intervention,
            Some(Intervention::Reappraisal { .. })
        ));
        assert_eq!(result.text_complexity, TextComplexity::Simplified);
        assert_eq!(result.modality, Modality::Visual);
        assert!((result.difficulty_delta - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn frustration_high_stress_triggers_grounding() {
        let state = CognitiveState {
            valence: -0.8,
            arousal: 0.9,
            stress: 0.7, // above grounding threshold
            ..default_state()
        };
        let result = adapt_content(&state, "fractions", 400, 0.1, 5, 5, &[]);
        assert!(matches!(
            result.intervention,
            Some(Intervention::Grounding { .. })
        ));
    }

    #[test]
    fn frustration_below_threshold_no_frustration_intervention() {
        // Positive valence -> frustration score near 0.
        let state = CognitiveState {
            valence: 0.3,
            arousal: 0.4,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt_content(&state, "addition", 500, 0.6, 1, 5, &[]);
        // No frustration-based intervention (might still get other interventions)
        assert_eq!(result.text_complexity, TextComplexity::Standard);
        // Should NOT be the frustration path
        assert!(result.reasoning.contains("Phi="));
        assert!(!result.reasoning.contains("Frustration detected"));
    }

    #[test]
    fn frustration_zero_failures_no_trigger() {
        // Even with negative valence and high arousal, zero failures -> score = 0.
        let state = CognitiveState {
            valence: -0.9,
            arousal: 0.95,
            ..default_state()
        };
        let result = adapt_content(&state, "addition", 500, 0.8, 0, 5, &[]);
        assert!(!result.reasoning.contains("Frustration detected"));
    }

    // -----------------------------------------------------------------------
    // Text simplification
    // -----------------------------------------------------------------------

    #[test]
    fn simplify_replaces_complex_words() {
        let text = "Sam purchased 5 bags. How many remaining?";
        let simplified = simplify_text(text, 3);
        assert!(simplified.contains("bought"));
        assert!(simplified.contains("left"));
        assert!(!simplified.contains("purchased"));
        assert!(!simplified.contains("remaining"));
    }

    #[test]
    fn simplify_grade_dependent_rectangle() {
        // Grade 3 (ordinal 4): rectangle -> shape
        let early = simplify_text("Draw a rectangle.", 4);
        assert!(early.contains("shape"));

        // Grade 5 (ordinal 6): rectangle stays
        let later = simplify_text("Draw a rectangle.", 6);
        assert!(later.contains("rectangle"));
    }

    #[test]
    fn simplify_grade_dependent_multiplication() {
        // Grade 2 (ordinal 3): multiplication -> times
        let early = simplify_text("Use multiplication.", 3);
        assert!(early.contains("times"));

        // Grade 4 (ordinal 5): multiplication stays
        let later = simplify_text("Use multiplication.", 5);
        assert!(later.contains("multiplication"));
    }

    // -----------------------------------------------------------------------
    // Personalization
    // -----------------------------------------------------------------------

    #[test]
    fn personalize_dinosaurs_replaces_items() {
        let text = "Sam has 3 bags with 4 apples each.";
        let result = personalize_text(text, "dinosaurs");
        assert!(result.contains("nests"));
        assert!(result.contains("dinosaur eggs"));
        assert!(!result.contains("bags"));
        assert!(!result.contains("apples"));
    }

    #[test]
    fn personalize_minecraft() {
        let text = "There are 5 boxes of stickers.";
        let result = personalize_text(text, "minecraft");
        assert!(result.contains("chests"));
        assert!(result.contains("diamonds"));
    }

    #[test]
    fn personalize_unknown_interest_uses_raw() {
        let text = "There are 3 groups of items.";
        let result = personalize_text(text, "robots");
        assert!(result.contains("groups")); // subject stays "groups"
        assert!(result.contains("robots")); // item becomes interest
    }

    // -----------------------------------------------------------------------
    // Equation extraction
    // -----------------------------------------------------------------------

    #[test]
    fn extract_equation_from_word_problem() {
        let text = "Sam has 3 + 4 apples. How many?";
        let eq = extract_equation(text);
        assert!(eq.contains("3 + 4"));
        assert!(!eq.contains("Sam"));
    }

    #[test]
    fn extract_equation_no_operators_takes_first_sentence() {
        let text = "Sam has some apples. He gave them away.";
        let eq = extract_equation(text);
        assert_eq!(eq, "Sam has some apples");
    }

    // -----------------------------------------------------------------------
    // Stress-triggered break
    // -----------------------------------------------------------------------

    #[test]
    fn high_stress_triggers_mandatory_break() {
        let state = CognitiveState {
            stress: 0.7,
            valence: 0.1, // not frustrated (positive valence)
            arousal: 0.3,
            ..default_state()
        };
        let result = adapt_content(&state, "algebra", 500, 0.5, 0, 8, &[]);
        assert!(matches!(
            result.intervention,
            Some(Intervention::Break { .. })
        ));
    }

    #[test]
    fn high_stress_young_student_gets_longer_break() {
        let state = CognitiveState {
            stress: 0.7,
            valence: 0.1,
            arousal: 0.3,
            ..default_state()
        };
        // PreK (grade_ordinal 0) -> 10 min break
        let result = adapt_content(&state, "counting", 500, 0.5, 0, 0, &[]);
        if let Some(Intervention::Break {
            duration_minutes, ..
        }) = &result.intervention
        {
            assert_eq!(*duration_minutes, 10);
        } else {
            panic!("Expected Break intervention for young student");
        }
    }

    #[test]
    fn high_stress_older_student_gets_shorter_break() {
        let state = CognitiveState {
            stress: 0.7,
            valence: 0.1,
            arousal: 0.3,
            ..default_state()
        };
        // Grade 5 (ordinal 6) -> 5 min break
        let result = adapt_content(&state, "decimals", 500, 0.5, 0, 6, &[]);
        if let Some(Intervention::Break {
            duration_minutes, ..
        }) = &result.intervention
        {
            assert_eq!(*duration_minutes, 5);
        } else {
            panic!("Expected Break intervention for older student");
        }
    }

    // -----------------------------------------------------------------------
    // Peer suggestion
    // -----------------------------------------------------------------------

    #[test]
    fn high_social_readiness_with_failures_suggests_peer() {
        let state = CognitiveState {
            social_readiness: 0.7,
            valence: 0.1, // slightly positive -> frustration_score low
            arousal: 0.3,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt_content(&state, "geometry", 300, 0.2, 4, 5, &[]);
        assert!(result.peer_suggestion.is_some());
        let peer = result.peer_suggestion.unwrap();
        assert!(peer.peer_criteria.needs_patience);
        assert!(peer.peer_criteria.compatible_affect);
        assert_eq!(peer.peer_criteria.needs_high_mastery_in, "geometry");
    }

    #[test]
    fn low_social_readiness_no_peer_suggestion() {
        let state = CognitiveState {
            social_readiness: 0.3, // below 0.6 threshold
            ..default_state()
        };
        let result = adapt_content(&state, "geometry", 300, 0.2, 4, 5, &[]);
        assert!(result.peer_suggestion.is_none());
    }

    #[test]
    fn few_failures_no_peer_suggestion() {
        let state = CognitiveState {
            social_readiness: 0.7,
            ..default_state()
        };
        // Only 1 failure (need >= 3)
        let result = adapt_content(&state, "geometry", 300, 0.6, 1, 5, &[]);
        assert!(result.peer_suggestion.is_none());
    }

    // -----------------------------------------------------------------------
    // FEP motor command -> difficulty
    // -----------------------------------------------------------------------

    #[test]
    fn exploration_trigger_increases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::ExplorationTrigger,
            permeability: 1.0, // full range
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(
            result.difficulty_delta > 0.0,
            "ExplorationTrigger should increase difficulty, got {}",
            result.difficulty_delta
        );
    }

    #[test]
    fn memory_consolidate_decreases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::MemoryConsolidate,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(
            result.difficulty_delta < 0.0,
            "MemoryConsolidate should decrease difficulty, got {}",
            result.difficulty_delta
        );
    }

    #[test]
    fn learning_rate_adjust_decreases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::LearningRateAdjust,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta < 0.0);
    }

    #[test]
    fn reflection_initiate_zero_difficulty_change() {
        let state = CognitiveState {
            motor_command: MotorCommand::ReflectionInitiate,
            permeability: 1.0,
            stress: 0.2, // low stress so we don't trigger break
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(
            result.difficulty_delta.abs() < f32::EPSILON,
            "ReflectionInitiate should not change difficulty"
        );
    }

    // -----------------------------------------------------------------------
    // Free energy -> difficulty (NoOp motor command)
    // -----------------------------------------------------------------------

    #[test]
    fn noop_high_free_energy_decreases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::NoOp,
            free_energy: 0.8, // high surprise = too hard
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.5, 0, 5, &[]);
        assert!(
            result.difficulty_delta < 0.0,
            "High free energy (too hard) should decrease difficulty"
        );
    }

    #[test]
    fn noop_low_free_energy_increases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::NoOp,
            free_energy: 0.2, // low surprise = too easy
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(
            result.difficulty_delta > 0.0,
            "Low free energy (too easy) should increase difficulty"
        );
    }

    #[test]
    fn noop_moderate_free_energy_no_change() {
        let state = CognitiveState {
            motor_command: MotorCommand::NoOp,
            free_energy: 0.5, // in the sweet spot
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(
            result.difficulty_delta.abs() < f32::EPSILON,
            "Moderate free energy should not change difficulty"
        );
    }

    // -----------------------------------------------------------------------
    // Permeability scaling
    // -----------------------------------------------------------------------

    #[test]
    fn low_permeability_narrows_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::ExplorationTrigger, // +0.3 raw
            permeability: 0.2, // very closed
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        // 0.3 * 0.2 = 0.06
        assert!(
            result.difficulty_delta < 0.1,
            "Low permeability should narrow the difficulty delta, got {}",
            result.difficulty_delta
        );
        assert!(result.difficulty_delta > 0.0); // still positive, just smaller
    }

    #[test]
    fn high_permeability_full_difficulty_range() {
        let state = CognitiveState {
            motor_command: MotorCommand::ExplorationTrigger, // +0.3 raw
            permeability: 1.0, // fully open
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(
            (result.difficulty_delta - 0.3).abs() < f32::EPSILON,
            "Full permeability should give full difficulty delta"
        );
    }

    // -----------------------------------------------------------------------
    // Modality selection
    // -----------------------------------------------------------------------

    #[test]
    fn high_focus_selects_text_modality() {
        let state = CognitiveState {
            focus: 0.8,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::Text);
    }

    #[test]
    fn high_social_readiness_selects_kinesthetic() {
        let state = CognitiveState {
            focus: 0.5,             // not high enough for Text
            social_readiness: 0.7,  // triggers Kinesthetic
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::Kinesthetic);
    }

    #[test]
    fn low_arousal_selects_visual() {
        let state = CognitiveState {
            focus: 0.4,
            social_readiness: 0.3,
            arousal: 0.2, // low arousal
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::Visual);
    }

    #[test]
    fn moderate_everything_selects_multimodal() {
        let state = CognitiveState {
            focus: 0.5,
            social_readiness: 0.4,
            arousal: 0.5,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::MultiModal);
    }

    // -----------------------------------------------------------------------
    // Challenge intervention
    // -----------------------------------------------------------------------

    #[test]
    fn high_phi_high_mastery_triggers_challenge() {
        let state = CognitiveState {
            phi: 0.7,
            stress: 0.2, // low stress so break doesn't trigger first
            motor_command: MotorCommand::NoOp,
            ..default_state()
        };
        let result = adapt_content(&state, "algebra", 900, 0.9, 0, 8, &[]);
        assert!(matches!(
            result.intervention,
            Some(Intervention::Challenge { .. })
        ));
    }

    #[test]
    fn reflection_motor_triggers_challenge() {
        let state = CognitiveState {
            motor_command: MotorCommand::ReflectionInitiate,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(matches!(
            result.intervention,
            Some(Intervention::Challenge { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Text complexity selection
    // -----------------------------------------------------------------------

    #[test]
    fn failures_with_interests_triggers_personalized() {
        let interests = vec!["dinosaurs".to_string()];
        let state = CognitiveState {
            focus: 0.5,
            ..default_state()
        };
        let result =
            adapt_content(&state, "multiplication", 300, 0.2, 3, 5, &interests);
        assert!(matches!(
            result.text_complexity,
            TextComplexity::Personalized { .. }
        ));
    }

    #[test]
    fn failures_no_interests_triggers_simplified() {
        let state = CognitiveState {
            focus: 0.5,
            ..default_state()
        };
        let result = adapt_content(&state, "multiplication", 300, 0.2, 3, 5, &[]);
        assert_eq!(result.text_complexity, TextComplexity::Simplified);
    }

    #[test]
    fn low_focus_triggers_minimal() {
        let state = CognitiveState {
            focus: 0.2, // below 0.3 threshold
            ..default_state()
        };
        // No consecutive failures -> not Simplified path
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.text_complexity, TextComplexity::Minimal);
    }

    #[test]
    fn normal_state_keeps_standard() {
        let state = CognitiveState {
            focus: 0.5,
            ..default_state()
        };
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.text_complexity, TextComplexity::Standard);
    }

    // -----------------------------------------------------------------------
    // Grounding exercises
    // -----------------------------------------------------------------------

    #[test]
    fn young_student_high_stress_gets_sensory_grounding() {
        let exercise = grounding_exercise(3, 0.85);
        if let Intervention::Grounding {
            duration_seconds,
            exercise: text,
        } = &exercise
        {
            assert_eq!(*duration_seconds, 90);
            assert!(text.contains("5-4-3-2-1"));
        } else {
            panic!("Expected Grounding intervention for young + high stress");
        }
    }

    #[test]
    fn young_student_moderate_stress_gets_breathing() {
        let exercise = grounding_exercise(2, 0.5);
        if let Intervention::Grounding {
            duration_seconds,
            exercise: text,
        } = &exercise
        {
            assert_eq!(*duration_seconds, 60);
            assert!(text.contains("breaths"));
        } else {
            panic!("Expected Grounding for young + moderate stress");
        }
    }

    #[test]
    fn older_student_gets_reappraisal() {
        let exercise = grounding_exercise(8, 0.85);
        assert!(matches!(exercise, Intervention::Reappraisal { .. }));
    }

    // -----------------------------------------------------------------------
    // Signal derivation
    // -----------------------------------------------------------------------

    #[test]
    fn derive_cognitive_state_cortisol() {
        let state =
            derive_cognitive_state(0.5, 0.5, 0.5, 0.2, 0.8, MotorCommand::NoOp);
        // cortisol = NE * (1 - 5HT) = 0.8 * 0.8 = 0.64
        assert!((state.stress - 0.64).abs() < 0.01);
    }

    #[test]
    fn derive_cognitive_state_valence() {
        let state =
            derive_cognitive_state(0.5, 0.5, 0.3, 0.2, 0.8, MotorCommand::NoOp);
        // cortisol = 0.8 * (1 - 0.2) = 0.64
        // valence = DA - cortisol = 0.3 - 0.64 = -0.34
        assert!((state.valence - (-0.34)).abs() < 0.01);
    }

    #[test]
    fn derive_cognitive_state_permeability_sigmoid() {
        // High serotonin + low NE -> high permeability
        let open = derive_cognitive_state(0.5, 0.5, 0.5, 0.8, 0.1, MotorCommand::NoOp);
        assert!(
            open.permeability > 0.7,
            "High 5HT + low NE should give high permeability, got {}",
            open.permeability
        );

        // Low serotonin + high NE -> low permeability
        let closed =
            derive_cognitive_state(0.5, 0.5, 0.3, 0.1, 0.9, MotorCommand::NoOp);
        assert!(
            closed.permeability < 0.4,
            "Low 5HT + high NE should give low permeability, got {}",
            closed.permeability
        );
    }

    #[test]
    fn derive_cognitive_state_oxytocin() {
        let state =
            derive_cognitive_state(0.5, 0.5, 0.6, 0.8, 0.3, MotorCommand::NoOp);
        // oxytocin = 5HT * 0.6 + DA * 0.4 = 0.48 + 0.24 = 0.72
        assert!((state.social_readiness - 0.72).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Rewrite integration
    // -----------------------------------------------------------------------

    #[test]
    fn rewrite_standard_returns_original() {
        let text = "Sam has 5 apples.";
        assert_eq!(rewrite_problem(text, &TextComplexity::Standard, 5), text);
    }

    #[test]
    fn rewrite_simplified_transforms_text() {
        let text = "Sam purchased 3 items.";
        let result = rewrite_problem(text, &TextComplexity::Simplified, 3);
        assert!(result.contains("bought"));
    }

    #[test]
    fn rewrite_minimal_extracts_equation() {
        let text = "Sam has 3 + 4 apples.";
        let result = rewrite_problem(text, &TextComplexity::Minimal, 5);
        assert!(result.contains("3 + 4"));
        assert!(!result.contains("Sam"));
    }

    #[test]
    fn rewrite_personalized_uses_interest() {
        let text = "There are 3 bags of apples.";
        let complexity = TextComplexity::Personalized {
            interest_topic: "space".to_string(),
        };
        let result = rewrite_problem(text, &complexity, 5);
        assert!(result.contains("rockets"));
        assert!(result.contains("astronauts"));
    }

    // -----------------------------------------------------------------------
    // Reasoning trace
    // -----------------------------------------------------------------------

    #[test]
    fn reasoning_trace_includes_all_signals() {
        let state = default_state();
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        assert!(result.reasoning.contains("Phi="));
        assert!(result.reasoning.contains("FE="));
        assert!(result.reasoning.contains("valence="));
        assert!(result.reasoning.contains("arousal="));
        assert!(result.reasoning.contains("stress="));
        assert!(result.reasoning.contains("permeability="));
        assert!(result.reasoning.contains("Motor:"));
        assert!(result.reasoning.contains("Mastery:"));
        assert!(result.reasoning.contains("Accuracy:"));
        assert!(result.reasoning.contains("Failures:"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn all_zeros_state_does_not_panic() {
        let state = CognitiveState {
            phi: 0.0,
            free_energy: 0.0,
            valence: 0.0,
            arousal: 0.0,
            focus: 0.0,
            motivation: 0.0,
            stress: 0.0,
            social_readiness: 0.0,
            permeability: 0.0,
            motor_command: MotorCommand::NoOp,
        };
        let result = adapt_content(&state, "math", 0, 0.0, 0, 0, &[]);
        // Should not panic, difficulty should be 0 (scaled by 0 permeability)
        assert!(result.difficulty_delta.abs() < f32::EPSILON);
    }

    #[test]
    fn all_max_state_does_not_panic() {
        let state = CognitiveState {
            phi: 1.0,
            free_energy: 1.5,
            valence: 1.0,
            arousal: 1.0,
            focus: 1.0,
            motivation: 1.0,
            stress: 1.0,
            social_readiness: 1.0,
            permeability: 1.0,
            motor_command: MotorCommand::ExplorationTrigger,
        };
        let result = adapt_content(&state, "math", 1000, 1.0, 100, 13, &[]);
        // High stress -> break takes priority
        assert!(matches!(
            result.intervention,
            Some(Intervention::Break { .. })
        ));
    }

    #[test]
    fn serialization_roundtrip() {
        let state = default_state();
        let result = adapt_content(&state, "math", 500, 0.8, 0, 5, &[]);
        let json = serde_json::to_string(&result).expect("serialize");
        let _: ContentAdaptation =
            serde_json::from_str(&json).expect("deserialize");
    }
}
