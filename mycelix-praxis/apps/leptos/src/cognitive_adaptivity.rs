// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Sovereignty-respecting cognitive adaptivity engine for Praxis.
//!
//! This module adapts learning content in real-time based on the student's
//! cognitive state. It **offers, never imposes**. Every suggestion includes
//! a "no thanks" option, and the student's decline is respected without
//! consequence.
//!
//! # Sovereignty model
//!
//! Students earn autonomy through demonstrated self-regulation, not by age
//! alone. The system operates in four modes:
//!
//! - **Guardian**: system adapts content and explains what it did.
//! - **Guide**: system suggests options, student chooses.
//! - **Mirror**: system shows cognitive state, student decides.
//! - **Autonomous**: system is available on request only.
//!
//! Sovereignty **never decreases** from wrong answers or poor choices.
//! Failure is data, not punishment.
//!
//! # Safety threshold
//!
//! The ONE exception to pure suggestion: a configurable safety threshold
//! set by the teacher or parent (never by the system itself). Even then,
//! the student sees who set it and can say "I'm OK".
//!
//! # Signal derivation
//!
//! From the three Spore neuromodulators (DA, 5-HT, NE) we derive:
//! - **Cortisol proxy**: NE * (1 - 5HT) -- stress
//! - **Acetylcholine proxy**: consciousness_level -- focused attention
//! - **Oxytocin proxy**: 5HT * 0.6 + DA * 0.4 -- social bonding readiness
//! - **Valence**: DA - cortisol -- frustrated (-1) to happy (+1)
//! - **Arousal**: NE + DA * 0.5 -- calm (0) to excited (1)
//! - **Markov permeability**: sigmoid(5HT + oxytocin - NE - cortisol)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sovereignty system
// ---------------------------------------------------------------------------

/// Sovereignty level — the student's relationship with the system.
///
/// # Philosophy: Trust by default
///
/// Sovereignty starts HIGH (600 — Mirror mode). The system must earn the
/// right to guide, not the child earn the right to be free. The student
/// begins as a trusted agent with the system available as a tool they
/// pick up when they want it.
///
/// The system can offer more structure if the student *asks* for help
/// or if a teacher/parent has configured a support level. But it never
/// imposes structure uninvited.
///
/// Sovereignty still never decreases from failure — failure is data,
/// not punishment.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SovereigntyLevel {
    /// Overall sovereignty (0-1000 permille).
    /// 0-200: Guardian mode (system leads, explains)
    /// 201-500: Guide mode (system suggests, student chooses)
    /// 501-800: Mirror mode (system shows state, student decides)
    /// 801-1000: Autonomous (system available on request)
    pub level: u16,

    /// How sovereignty was earned (audit trail for transparency).
    pub growth_events: Vec<SovereigntyEvent>,

    /// Whether the student has entered sandbox mode (no tracking).
    pub sandbox_active: bool,

    /// Whether the student has explicitly requested more support.
    /// When true, the system temporarily operates one mode lower
    /// than their earned level, until they dismiss the support.
    pub support_requested: bool,
}

/// A record of sovereignty growth, visible to student and teacher.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SovereigntyEvent {
    pub event_type: SovereigntyGrowthType,
    pub description: String,
    /// Always non-negative in practice; i16 for schema flexibility.
    pub delta: i16,
    pub timestamp: u64,
}

/// Ways a student demonstrates self-regulation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SovereigntyGrowthType {
    /// Chose to take a break when stress was rising.
    SelfRegulatedBreak,
    /// Pushed through difficulty and succeeded.
    PerseveranceSuccess,
    /// Asked for help when stuck (knowing when to ask IS self-regulation).
    AskedForHelp,
    /// Made a learning plan and followed it.
    PlannedAndExecuted,
    /// Reflected on a mistake constructively.
    ReflectiveResponse,
    /// Chose appropriate difficulty for themselves.
    DifficultyCalibration,
    /// Helped a peer learn.
    PeerTeaching,
    /// Overrode a system suggestion and it worked out.
    IndependentSuccess,
    /// Fulfilled a reciprocity pledge (tutoring, translation, curriculum review).
    FulfilledPledge,
    /// Made a community contribution (non-teaching: funding, content, review).
    CommunityContribution,
}

impl SovereigntyLevel {
    /// Create a new sovereignty level. Trust by default — starts at 600
    /// (Mirror mode). The system is a tool the child picks up, not a
    /// monitor they can't turn off.
    pub fn new() -> Self {
        Self {
            level: 600, // Trust by default: Mirror mode
            growth_events: Vec::new(),
            sandbox_active: false,
            support_requested: false,
        }
    }

    /// What interaction mode should the system use?
    ///
    /// If the student has requested support, operates one mode lower
    /// than their earned level — because they asked, not because the
    /// system decided.
    pub fn mode(&self) -> InteractionMode {
        if self.sandbox_active {
            return InteractionMode::Autonomous; // No tracking in sandbox
        }

        let effective = if self.support_requested {
            self.level.saturating_sub(300) // One mode lower
        } else {
            self.level
        };

        match effective {
            0..=200 => InteractionMode::Guardian,
            201..=500 => InteractionMode::Guide,
            501..=800 => InteractionMode::Mirror,
            _ => InteractionMode::Autonomous,
        }
    }

    /// Enter sandbox mode — exploration without measurement.
    /// No mastery tracking, no sovereignty events, no adaptivity.
    /// Just content to explore. The child can leave at any time.
    pub fn enter_sandbox(&mut self) {
        self.sandbox_active = true;
    }

    /// Leave sandbox mode — return to normal tracking.
    pub fn leave_sandbox(&mut self) {
        self.sandbox_active = false;
    }

    /// Student explicitly requests more structure/support.
    /// This is NOT the system imposing — the student chose this.
    pub fn request_support(&mut self) {
        self.support_requested = true;
    }

    /// Student dismisses extra support, returns to earned level.
    pub fn dismiss_support(&mut self) {
        self.support_requested = false;
    }

    /// Record a positive sovereignty event.
    /// Sovereignty NEVER decreases -- failure is data, not punishment.
    pub fn record_growth(
        &mut self,
        event_type: SovereigntyGrowthType,
        description: String,
        timestamp: u64,
    ) {
        let delta = match &event_type {
            SovereigntyGrowthType::SelfRegulatedBreak => 15,
            SovereigntyGrowthType::PerseveranceSuccess => 10,
            SovereigntyGrowthType::AskedForHelp => 12,
            SovereigntyGrowthType::PlannedAndExecuted => 20,
            SovereigntyGrowthType::ReflectiveResponse => 15,
            SovereigntyGrowthType::DifficultyCalibration => 10,
            SovereigntyGrowthType::PeerTeaching => 25,
            SovereigntyGrowthType::IndependentSuccess => 20,
            SovereigntyGrowthType::FulfilledPledge => 20,
            SovereigntyGrowthType::CommunityContribution => 15,
        };

        self.level = (self.level + delta as u16).min(1000);
        self.growth_events.push(SovereigntyEvent {
            event_type,
            description,
            delta,
            timestamp,
        });
    }
}

/// Interaction mode determines how the system surfaces information.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InteractionMode {
    /// System leads, explains after. Only when the student *asks* for help
    /// or a teacher/parent configures this level.
    Guardian,
    /// System suggests, student chooses.
    Guide,
    /// System shows learning state, student decides. The default starting mode.
    /// Uses honest language: "engagement estimate" not "consciousness level."
    Mirror,
    /// System available on request only. No unsolicited anything.
    Autonomous,
}

impl InteractionMode {
    /// Honest, kid-friendly label for this mode.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Guardian => "Helper",
            Self::Guide => "Guide",
            Self::Mirror => "Mirror",
            Self::Autonomous => "Independent",
        }
    }

    /// Description that explains the relationship honestly.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Guardian => "I'll help guide you and explain what I'm doing. You asked for extra support — you can stop anytime.",
            Self::Guide => "I'll suggest options when you might want them. You always choose.",
            Self::Mirror => "I'll show you how your learning is going. You decide what to do.",
            Self::Autonomous => "You're in charge. I'm here if you need me.",
        }
    }
}

// ---------------------------------------------------------------------------
// Honest measurement language
// ---------------------------------------------------------------------------

/// What we actually measure, described honestly.
/// We don't call it "consciousness" — we call it what it is.
pub struct HonestMetrics {
    pub engagement_estimate: f32,       // was "Phi" / "consciousness level"
    pub confusion_estimate: f32,        // was "free energy"
    pub mood_estimate: f32,             // was "valence"
    pub energy_estimate: f32,           // was "arousal"
    pub focus_estimate: f32,            // was "acetylcholine proxy"
    pub motivation_estimate: f32,       // was "dopamine"
    pub stress_estimate: f32,           // was "cortisol proxy"
}

impl HonestMetrics {
    /// Convert from internal CognitiveState to honest labels.
    pub fn from_cognitive_state(state: &CognitiveState) -> Self {
        Self {
            engagement_estimate: state.phi,
            confusion_estimate: state.free_energy,
            mood_estimate: state.valence,
            energy_estimate: state.arousal,
            focus_estimate: state.focus,
            motivation_estimate: state.motivation,
            stress_estimate: state.stress,
        }
    }

    /// Kid-friendly summary: one sentence about how they're doing.
    pub fn summary(&self) -> &'static str {
        if self.stress_estimate > 0.6 {
            "You seem a bit stressed. It's OK to take a break if you want."
        } else if self.engagement_estimate > 0.6 && self.focus_estimate > 0.5 {
            "You're focused and doing great work."
        } else if self.motivation_estimate < 0.3 {
            "Feeling low energy? That's normal. You could try something different, or come back later."
        } else if self.confusion_estimate > 0.7 {
            "This is tricky! Being confused means your brain is working hard."
        } else {
            "You're doing fine. Keep going at your own pace."
        }
    }

    /// Disclaimer shown to parents/teachers.
    pub fn measurement_disclaimer() -> &'static str {
        "These are estimates based on interaction patterns, not direct brain \
         measurements. They help personalize the experience but are not \
         diagnostic. If you have concerns about your child's learning, \
         please consult an educator or specialist."
    }
}

// ---------------------------------------------------------------------------
// Curiosity-driven exploration
// ---------------------------------------------------------------------------

/// A curiosity prompt — asks what the student wants to explore.
/// The curriculum is a map, not a road.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CuriosityPrompt {
    pub question: String,
    pub options: Vec<CuriosityOption>,
    pub free_text_enabled: bool,
}

/// An option in the curiosity prompt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CuriosityOption {
    pub label: String,
    pub topic: String,
    pub icon: String,
}

/// Generate a curiosity prompt based on what the student has been learning.
/// Shows related topics they could explore — or lets them type anything.
pub fn curiosity_prompt(
    recent_topics: &[String],
    grade_ordinal: u8,
) -> CuriosityPrompt {
    let mut options = Vec::new();

    // Suggest related but unexplored areas
    if recent_topics.iter().any(|t| t.contains("Multiplication")) {
        options.push(CuriosityOption {
            label: "How does multiplication work with really big numbers?".into(),
            topic: "Multiplication".into(),
            icon: "\u{1f522}".into(), // 🔢
        });
        options.push(CuriosityOption {
            label: "What are patterns in the times tables?".into(),
            topic: "Patterns".into(),
            icon: "\u{1f3b5}".into(), // 🎵
        });
    }

    if recent_topics.iter().any(|t| t.contains("Fractions")) {
        options.push(CuriosityOption {
            label: "Why do we need fractions in real life?".into(),
            topic: "Fractions".into(),
            icon: "\u{1f355}".into(), // 🍕
        });
    }

    if recent_topics.iter().any(|t| t.contains("Geometry")) {
        options.push(CuriosityOption {
            label: "What shapes can you find in buildings?".into(),
            topic: "Geometry".into(),
            icon: "\u{1f3db}\u{fe0f}".into(), // 🏛️
        });
    }

    // Always offer open-ended exploration
    if grade_ordinal <= 5 {
        options.push(CuriosityOption {
            label: "I want to explore something totally different!".into(),
            topic: "explore".into(),
            icon: "\u{1f30d}".into(), // 🌍
        });
    } else {
        options.push(CuriosityOption {
            label: "I have my own question".into(),
            topic: "explore".into(),
            icon: "\u{2753}".into(), // ❓
        });
    }

    CuriosityPrompt {
        question: "What are you curious about?".into(),
        options,
        free_text_enabled: true,
    }
}

// ---------------------------------------------------------------------------
// Suggestion system (replaces mandatory interventions)
// ---------------------------------------------------------------------------

/// A suggestion the student can accept, modify, or decline.
/// The system NEVER forces an action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    pub suggestion_type: SuggestionType,
    pub message: String,
    pub options: Vec<StudentChoice>,
    /// Why the system suggested this (visible on teacher dashboard).
    pub reasoning_for_teacher: String,
    /// Only true when a teacher/parent-configured safety threshold is exceeded.
    pub is_safety_concern: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SuggestionType {
    TakeBreak,
    TryDifferentApproach,
    SimplifyContent,
    WorkWithPeer,
    TryHarderContent,
    Reflect,
    SwitchTopic,
    CelebrateProgress,
}

/// An option presented to the student. There is ALWAYS a "no thanks" option.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StudentChoice {
    pub label: String,
    pub action: ChoiceAction,
    /// Marks the "no thanks" option.
    pub is_decline: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ChoiceAction {
    AcceptSuggestion,
    KeepGoing,
    TakeBreak { minutes: u32 },
    SimplifyProblem,
    SwitchToVisual,
    WorkWithPeer,
    /// "I want to do something else."
    ChooseOwn,
}

// ---------------------------------------------------------------------------
// Core types (preserved from original)
// ---------------------------------------------------------------------------

/// Real-time cognitive state derived from consciousness + neuromodulators.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveState {
    pub phi: f32,
    pub free_energy: f32,
    pub valence: f32,
    pub arousal: f32,
    pub focus: f32,
    pub motivation: f32,
    pub stress: f32,
    pub social_readiness: f32,
    pub permeability: f32,
    pub motor_command: MotorCommand,
}

/// FEP motor command mapped to educational actions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MotorCommand {
    AttentionShift,
    LearningRateAdjust,
    ExplorationTrigger,
    ReflectionInitiate,
    MemoryConsolidate,
    ExpectationReset,
    MotorOutput,
    NoOp,
}

// ---------------------------------------------------------------------------
// Adaptation output types
// ---------------------------------------------------------------------------

/// Content adaptation decisions produced by the engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContentAdaptation {
    pub text_complexity: TextComplexity,
    pub modality: Modality,
    pub difficulty_delta: f32,
    /// Sovereignty-respecting suggestion (replaces mandatory interventions).
    pub suggestion: Option<Suggestion>,
    pub peer_suggestion: Option<PeerSuggestion>,
    pub reasoning: String,
}

/// Text complexity levels for content rewriting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TextComplexity {
    Standard,
    Simplified,
    Minimal,
    Personalized { interest_topic: String },
}

/// Presentation modality.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Visual,
    Auditory,
    Kinesthetic,
    MultiModal,
}

/// Suggestion for peer pairing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeerSuggestion {
    pub reason: String,
    pub peer_criteria: PeerCriteria,
    pub duration_minutes: u32,
    pub expected_benefit: String,
}

/// Criteria for selecting a compatible peer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeerCriteria {
    pub needs_high_mastery_in: String,
    pub needs_patience: bool,
    pub compatible_affect: bool,
}

/// Result of a sovereignty-aware rewrite decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RewriteResult {
    /// Guardian mode: rewrite applied, original available.
    Applied {
        rewritten: String,
        original: String,
        explanation: String,
    },
    /// Guide mode: rewrite offered, student chooses.
    Offered {
        rewritten: String,
        original: String,
        prompt: String,
    },
    /// Mirror/Autonomous: original shown, simplification available on request.
    Available { original: String },
}

// ---------------------------------------------------------------------------
// Signal derivation
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Build a `CognitiveState` from raw consciousness signals.
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
// Core decision engine (sovereignty-aware)
// ---------------------------------------------------------------------------

/// Default safety threshold. Teacher/parent can override per-student.
const DEFAULT_SAFETY_THRESHOLD: f32 = 0.85;

/// Compute content adaptation from cognitive state, respecting sovereignty.
///
/// The function returns mode-appropriate output:
/// - **Guardian**: adapts content AND suggests action with explanation.
/// - **Guide**: offers multiple paths, student chooses.
/// - **Mirror**: shows cognitive state data, no unsolicited suggestions.
/// - **Autonomous**: no suggestions unless student asks.
pub fn adapt_content(
    state: &CognitiveState,
    sovereignty: &SovereigntyLevel,
    current_skill: &str,
    current_mastery_permille: u16,
    recent_accuracy: f32,
    consecutive_failures: u32,
    grade_ordinal: u8,
    student_interests: &[String],
    safety_threshold: f32,
) -> ContentAdaptation {
    // === SANDBOX MODE: no adaptation, no tracking ===
    // The student chose to explore freely. Respect that completely.
    if sovereignty.sandbox_active {
        return ContentAdaptation {
            text_complexity: TextComplexity::Standard,
            modality: Modality::MultiModal,
            difficulty_delta: 0.0,
            suggestion: None,
            peer_suggestion: None,
            reasoning: "Sandbox mode — exploring freely, no adaptation".into(),
        };
    }

    let mode = sovereignty.mode();

    // === SAFETY CHECK (the ONE exception to pure suggestion) ===
    // Configured by teacher/parent, never by the system itself.
    if state.stress > safety_threshold {
        let break_mins = if grade_ordinal <= 1 { 10 } else { 5 };
        return ContentAdaptation {
            text_complexity: TextComplexity::Standard,
            modality: Modality::Visual,
            difficulty_delta: 0.0,
            suggestion: Some(Suggestion {
                suggestion_type: SuggestionType::TakeBreak,
                message: "Your teacher set a rest point here to help you stay \
                          healthy. Let's take a quick break."
                    .into(),
                options: vec![
                    StudentChoice {
                        label: "OK, I'll take a break".into(),
                        action: ChoiceAction::TakeBreak { minutes: break_mins },
                        is_decline: false,
                    },
                    StudentChoice {
                        label: "I'm really OK, let me finish this problem first"
                            .into(),
                        action: ChoiceAction::KeepGoing,
                        is_decline: true,
                    },
                ],
                reasoning_for_teacher: format!(
                    "Stress proxy {:.2} exceeded safety threshold {:.2}",
                    state.stress, safety_threshold
                ),
                is_safety_concern: true,
            }),
            peer_suggestion: None,
            reasoning: format!(
                "Safety threshold exceeded (stress {:.2} > {:.2}). Suggesting break.",
                state.stress, safety_threshold
            ),
        };
    }

    // === STRUGGLE CLASSIFICATION ===
    // Distinguish productive struggle (confused but engaged) from
    // distress (confused and withdrawing). Productive struggle is
    // where the deepest learning happens — leave them alone.
    let frustration_score = (-state.valence).max(0.0)
        * state.arousal
        * (consecutive_failures as f32 / 3.0).min(1.0);

    // Productive struggle: wrong answers but still engaged (high Phi, moderate arousal)
    // The system does NOT intervene here — this is where growth happens.
    let is_productive_struggle = consecutive_failures >= 2
        && state.phi > 0.4          // still engaged
        && state.arousal < 0.7      // not panicking
        && state.stress < 0.5;      // not distressed

    // === TEXT COMPLEXITY ===
    // Only simplify if the student is in distress, NOT productive struggle
    let text_complexity = if !is_productive_struggle && consecutive_failures >= 2 && recent_accuracy < 0.4 {
        if !student_interests.is_empty() {
            TextComplexity::Personalized {
                interest_topic: student_interests[0].clone(),
            }
        } else {
            TextComplexity::Simplified
        }
    } else if state.focus < 0.3 {
        TextComplexity::Minimal
    } else {
        TextComplexity::Standard
    };

    // === MODALITY FROM COGNITIVE STATE ===
    let modality = if state.focus > 0.7 {
        Modality::Text
    } else if state.social_readiness > 0.6 {
        Modality::Kinesthetic
    } else if state.arousal < 0.3 {
        Modality::Visual
    } else {
        Modality::MultiModal
    };

    // === DIFFICULTY FROM FEP MOTOR COMMAND ===
    let difficulty_delta = match state.motor_command {
        MotorCommand::ExplorationTrigger => 0.3,
        MotorCommand::LearningRateAdjust => -0.2,
        MotorCommand::ReflectionInitiate => 0.0,
        MotorCommand::MemoryConsolidate => -0.3,
        _ => {
            if state.free_energy > 0.7 {
                -0.3
            } else if state.free_energy < 0.3 {
                0.2
            } else {
                0.0
            }
        }
    };

    // === MARKOV PERMEABILITY -> ZPD MAPPING ===
    let effective_difficulty = difficulty_delta * state.permeability;

    // === SUGGESTION (sovereignty-aware) ===
    // During productive struggle, the system stays quiet. The student is
    // learning through difficulty — interrupting would rob them of that.
    let suggestion = if is_productive_struggle {
        None // Silence is support
    } else {
        build_suggestion(
            state,
            &mode,
            current_skill,
            current_mastery_permille,
            frustration_score,
            consecutive_failures,
            grade_ordinal,
        )
    };

    // === PEER SUGGESTION (always an offer, never an assignment) ===
    let peer_suggestion = if state.social_readiness > 0.6
        && consecutive_failures >= 3
        && frustration_score < 0.5
        && matches!(mode, InteractionMode::Guardian | InteractionMode::Guide)
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
            expected_benefit:
                "Students who explain concepts to peers deepen their own understanding"
                    .into(),
        })
    } else {
        None
    };

    // === REASONING TRACE ===
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
         Failures: {}. Mode: {:?}. -> {} complexity, {} modality, difficulty {}{:.1}",
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
        mode,
        complexity_label,
        modality_label,
        sign,
        effective_difficulty,
    );

    ContentAdaptation {
        text_complexity,
        modality,
        difficulty_delta: effective_difficulty,
        suggestion,
        peer_suggestion,
        reasoning,
    }
}

/// Build a sovereignty-appropriate suggestion based on cognitive state.
fn build_suggestion(
    state: &CognitiveState,
    mode: &InteractionMode,
    current_skill: &str,
    current_mastery_permille: u16,
    frustration_score: f32,
    consecutive_failures: u32,
    grade_ordinal: u8,
) -> Option<Suggestion> {
    // Autonomous mode: no unsolicited suggestions.
    if *mode == InteractionMode::Autonomous {
        return None;
    }

    // Mirror mode: only show data, no suggestions.
    if *mode == InteractionMode::Mirror {
        return None;
    }

    // High frustration: offer support (never force).
    if frustration_score > 0.7 {
        return Some(match mode {
            InteractionMode::Guardian => Suggestion {
                suggestion_type: SuggestionType::TryDifferentApproach,
                message: if state.stress > 0.65 {
                    grounding_message(grade_ordinal)
                } else {
                    format!(
                        "This is tricky! {} is a skill that takes practice. \
                         I'm going to make this a bit simpler.",
                        current_skill
                    )
                },
                options: vec![
                    StudentChoice {
                        label: "OK, sounds good".into(),
                        action: ChoiceAction::AcceptSuggestion,
                        is_decline: false,
                    },
                    StudentChoice {
                        label: "I want to keep trying the hard version".into(),
                        action: ChoiceAction::KeepGoing,
                        is_decline: true,
                    },
                ],
                reasoning_for_teacher: format!(
                    "Frustration score {:.2} exceeded threshold. Offering support.",
                    frustration_score
                ),
                is_safety_concern: false,
            },
            InteractionMode::Guide => Suggestion {
                suggestion_type: SuggestionType::TryDifferentApproach,
                message: format!(
                    "You've gotten the last {} wrong. What would help?",
                    consecutive_failures
                ),
                options: vec![
                    StudentChoice {
                        label: "Make the words simpler".into(),
                        action: ChoiceAction::SimplifyProblem,
                        is_decline: false,
                    },
                    StudentChoice {
                        label: "Show me a picture".into(),
                        action: ChoiceAction::SwitchToVisual,
                        is_decline: false,
                    },
                    StudentChoice {
                        label: "Let me try again".into(),
                        action: ChoiceAction::KeepGoing,
                        is_decline: true,
                    },
                    StudentChoice {
                        label: "Take a break".into(),
                        action: ChoiceAction::TakeBreak { minutes: 5 },
                        is_decline: false,
                    },
                    StudentChoice {
                        label: "Something else".into(),
                        action: ChoiceAction::ChooseOwn,
                        is_decline: false,
                    },
                ],
                reasoning_for_teacher: format!(
                    "Frustration score {:.2}. Offering choice of strategies.",
                    frustration_score
                ),
                is_safety_concern: false,
            },
            // Mirror and Autonomous handled above.
            _ => unreachable!(),
        });
    }

    // High consciousness + high mastery: celebrate and offer challenge.
    if state.phi > 0.6 && current_mastery_permille > 850 {
        return Some(Suggestion {
            suggestion_type: SuggestionType::CelebrateProgress,
            message: "You're really getting this! Ready for something harder?".into(),
            options: vec![
                StudentChoice {
                    label: "Yes, challenge me!".into(),
                    action: ChoiceAction::AcceptSuggestion,
                    is_decline: false,
                },
                StudentChoice {
                    label: "I want to practice more first".into(),
                    action: ChoiceAction::KeepGoing,
                    is_decline: true,
                },
            ],
            reasoning_for_teacher: format!(
                "Phi {:.2} with mastery {}\u{2030}. Offering challenge.",
                state.phi, current_mastery_permille
            ),
            is_safety_concern: false,
        });
    }

    // Reflection motor command.
    if state.motor_command == MotorCommand::ReflectionInitiate {
        return Some(Suggestion {
            suggestion_type: SuggestionType::Reflect,
            message: "Nice work! Can you explain how you solved that in your own words?"
                .into(),
            options: vec![
                StudentChoice {
                    label: "Sure, let me explain".into(),
                    action: ChoiceAction::AcceptSuggestion,
                    is_decline: false,
                },
                StudentChoice {
                    label: "I'd rather keep going".into(),
                    action: ChoiceAction::KeepGoing,
                    is_decline: true,
                },
            ],
            reasoning_for_teacher: "FEP motor command: ReflectionInitiate".into(),
            is_safety_concern: false,
        });
    }

    None
}

/// Age-appropriate grounding message (not forced -- offered).
fn grounding_message(grade_ordinal: u8) -> String {
    if grade_ordinal <= 4 {
        "Let's take 5 big breaths together! Breathe in like you're smelling \
         a flower... Breathe out like you're blowing out birthday candles..."
            .into()
    } else {
        "It's normal to find this challenging. Every expert was once a beginner. \
         Let's try breaking this down into smaller steps."
            .into()
    }
}

// ---------------------------------------------------------------------------
// Metacognitive prompts (sovereignty-aware)
// ---------------------------------------------------------------------------

/// Generate an optional metacognitive prompt based on sovereignty mode.
///
/// Respects sandbox mode (no prompts) and productive struggle (no
/// interruption when the student is engaged with difficulty).
pub fn metacognitive_prompt(
    sovereignty: &SovereigntyLevel,
    recent_result: bool,
    consecutive_correct: u32,
    skill: &str,
) -> Option<String> {
    // Sandbox: no prompts at all
    if sovereignty.sandbox_active {
        return None;
    }

    match sovereignty.mode() {
        InteractionMode::Guardian => {
            if consecutive_correct >= 3 {
                Some(format!(
                    "Great job! You got {} in a row! What's your trick for {}?",
                    consecutive_correct, skill
                ))
            } else if !recent_result {
                // Normalize struggle — don't treat wrong answers as problems
                Some(
                    "That was a tough one! Being stuck means your brain is \
                     working on it. Want to try again or try something different?"
                        .into(),
                )
            } else {
                None
            }
        }
        InteractionMode::Guide => Some(match (recent_result, consecutive_correct) {
            (true, n) if n >= 5 => {
                "You're on fire! Is this getting too easy? You can make it harder."
                    .into()
            }
            (true, _) => "Nice! How did you figure that out?".into(),
            (false, _) => {
                "Hmm, that didn't work. What would you try differently?".into()
            }
        }),
        // Mirror and Autonomous: no unsolicited prompts.
        // The student reflects on their own — we trust them.
        InteractionMode::Mirror | InteractionMode::Autonomous => None,
    }
}

// ---------------------------------------------------------------------------
// Word problem rewriter (sovereignty-aware)
// ---------------------------------------------------------------------------

/// Decide whether and how to rewrite a problem, respecting sovereignty.
pub fn suggest_rewrite(
    sovereignty: &SovereigntyLevel,
    original: &str,
    complexity: &TextComplexity,
    grade_ordinal: u8,
) -> RewriteResult {
    match sovereignty.mode() {
        InteractionMode::Guardian => RewriteResult::Applied {
            rewritten: rewrite_problem(original, complexity, grade_ordinal),
            original: original.to_string(),
            explanation:
                "I made the words simpler to help you focus on the math.".into(),
        },
        InteractionMode::Guide => RewriteResult::Offered {
            rewritten: rewrite_problem(original, complexity, grade_ordinal),
            original: original.to_string(),
            prompt: "Want me to make the words simpler?".into(),
        },
        InteractionMode::Mirror | InteractionMode::Autonomous => {
            RewriteResult::Available {
                original: original.to_string(),
            }
        }
    }
}

/// Rewrite a word problem based on text complexity.
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
pub fn simplify_text(text: &str, grade_ordinal: u8) -> String {
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
pub fn extract_equation(text: &str) -> String {
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
        text.split('.').next().unwrap_or(text).to_string()
    }
}

/// Replace context subjects with the student's interests.
pub fn personalize_text(text: &str, interest: &str) -> String {
    let generic_subjects = ["bags", "boxes", "groups", "baskets", "containers"];
    let generic_items = [
        "apples", "oranges", "items", "objects", "things", "stickers", "marbles",
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

/// Generate an age-appropriate grounding exercise.
pub fn grounding_exercise(grade_ordinal: u8, stress_level: f32) -> Suggestion {
    if grade_ordinal <= 4 {
        let (message, mins) = if stress_level > 0.8 {
            (
                "Let's do the 5-4-3-2-1 game! Name 5 things you can see, \
                 4 things you can touch, 3 things you can hear, \
                 2 things you can smell, and 1 thing you can taste.",
                3,
            )
        } else {
            (
                "Let's take 5 big breaths together! Breathe in like you're \
                 smelling a flower... Breathe out like you're blowing out \
                 birthday candles...",
                2,
            )
        };
        Suggestion {
            suggestion_type: SuggestionType::TakeBreak,
            message: message.into(),
            options: vec![
                StudentChoice {
                    label: "OK, let's do it".into(),
                    action: ChoiceAction::TakeBreak { minutes: mins },
                    is_decline: false,
                },
                StudentChoice {
                    label: "I'm fine, let me keep going".into(),
                    action: ChoiceAction::KeepGoing,
                    is_decline: true,
                },
            ],
            reasoning_for_teacher: format!(
                "Stress level {:.2}. Offering age-appropriate grounding.",
                stress_level
            ),
            is_safety_concern: false,
        }
    } else {
        Suggestion {
            suggestion_type: SuggestionType::TryDifferentApproach,
            message: "It's normal to find this challenging. Every expert was \
                      once a beginner. Let's try breaking this down into \
                      smaller steps."
                .into(),
            options: vec![
                StudentChoice {
                    label: "OK, break it down for me".into(),
                    action: ChoiceAction::AcceptSuggestion,
                    is_decline: false,
                },
                StudentChoice {
                    label: "I want to try again as-is".into(),
                    action: ChoiceAction::KeepGoing,
                    is_decline: true,
                },
            ],
            reasoning_for_teacher: format!(
                "Stress level {:.2}. Offering cognitive reappraisal.",
                stress_level
            ),
            is_safety_concern: false,
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
    // Helpers
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

    fn guardian() -> SovereigntyLevel {
        SovereigntyLevel::new() // level 100
    }

    fn guide() -> SovereigntyLevel {
        SovereigntyLevel {
            level: 350,
            growth_events: Vec::new(),
        }
    }

    fn mirror() -> SovereigntyLevel {
        SovereigntyLevel {
            level: 650,
            growth_events: Vec::new(),
        }
    }

    fn autonomous() -> SovereigntyLevel {
        SovereigntyLevel {
            level: 900,
            growth_events: Vec::new(),
        }
    }

    /// Shorthand for adapt_content with default safety threshold.
    fn adapt(
        state: &CognitiveState,
        sov: &SovereigntyLevel,
        skill: &str,
        mastery: u16,
        accuracy: f32,
        failures: u32,
        grade: u8,
        interests: &[String],
    ) -> ContentAdaptation {
        adapt_content(
            state,
            sov,
            skill,
            mastery,
            accuracy,
            failures,
            grade,
            interests,
            DEFAULT_SAFETY_THRESHOLD,
        )
    }

    // -----------------------------------------------------------------------
    // Sovereignty system
    // -----------------------------------------------------------------------

    #[test]
    fn sovereignty_starts_in_guardian() {
        let sov = SovereigntyLevel::new();
        assert_eq!(sov.level, 100);
        assert_eq!(sov.mode(), InteractionMode::Guardian);
    }

    #[test]
    fn sovereignty_mode_boundaries() {
        assert_eq!(
            SovereigntyLevel { level: 0, growth_events: vec![] }.mode(),
            InteractionMode::Guardian
        );
        assert_eq!(
            SovereigntyLevel { level: 200, growth_events: vec![] }.mode(),
            InteractionMode::Guardian
        );
        assert_eq!(
            SovereigntyLevel { level: 201, growth_events: vec![] }.mode(),
            InteractionMode::Guide
        );
        assert_eq!(
            SovereigntyLevel { level: 500, growth_events: vec![] }.mode(),
            InteractionMode::Guide
        );
        assert_eq!(
            SovereigntyLevel { level: 501, growth_events: vec![] }.mode(),
            InteractionMode::Mirror
        );
        assert_eq!(
            SovereigntyLevel { level: 800, growth_events: vec![] }.mode(),
            InteractionMode::Mirror
        );
        assert_eq!(
            SovereigntyLevel { level: 801, growth_events: vec![] }.mode(),
            InteractionMode::Autonomous
        );
        assert_eq!(
            SovereigntyLevel { level: 1000, growth_events: vec![] }.mode(),
            InteractionMode::Autonomous
        );
    }

    #[test]
    fn sovereignty_grows_from_self_regulation() {
        let mut sov = SovereigntyLevel::new();
        assert_eq!(sov.level, 100);
        sov.record_growth(
            SovereigntyGrowthType::SelfRegulatedBreak,
            "Chose to rest when stressed".into(),
            1000,
        );
        assert_eq!(sov.level, 115);
        assert_eq!(sov.growth_events.len(), 1);
        assert_eq!(sov.growth_events[0].delta, 15);
    }

    #[test]
    fn sovereignty_never_decreases_from_failure() {
        // Sovereignty has no decrease mechanism. This test documents the design.
        let mut sov = SovereigntyLevel::new();
        sov.record_growth(
            SovereigntyGrowthType::PerseveranceSuccess,
            "Kept going".into(),
            1000,
        );
        let level_after = sov.level;
        // No API to decrease -- level can only go up.
        assert!(level_after > 100);
    }

    #[test]
    fn sovereignty_caps_at_1000() {
        let mut sov = SovereigntyLevel {
            level: 995,
            growth_events: Vec::new(),
        };
        sov.record_growth(
            SovereigntyGrowthType::PeerTeaching,
            "Helped a peer".into(),
            1000,
        );
        assert_eq!(sov.level, 1000); // 995 + 25 = 1020, capped to 1000
    }

    #[test]
    fn sovereignty_growth_audit_trail() {
        let mut sov = SovereigntyLevel::new();
        sov.record_growth(
            SovereigntyGrowthType::AskedForHelp,
            "Asked teacher about fractions".into(),
            100,
        );
        sov.record_growth(
            SovereigntyGrowthType::ReflectiveResponse,
            "Explained what went wrong".into(),
            200,
        );
        assert_eq!(sov.growth_events.len(), 2);
        assert_eq!(
            sov.growth_events[0].event_type,
            SovereigntyGrowthType::AskedForHelp
        );
        assert_eq!(sov.growth_events[0].timestamp, 100);
        assert_eq!(
            sov.growth_events[1].event_type,
            SovereigntyGrowthType::ReflectiveResponse
        );
    }

    #[test]
    fn all_growth_types_have_positive_delta() {
        let types = [
            SovereigntyGrowthType::SelfRegulatedBreak,
            SovereigntyGrowthType::PerseveranceSuccess,
            SovereigntyGrowthType::AskedForHelp,
            SovereigntyGrowthType::PlannedAndExecuted,
            SovereigntyGrowthType::ReflectiveResponse,
            SovereigntyGrowthType::DifficultyCalibration,
            SovereigntyGrowthType::PeerTeaching,
            SovereigntyGrowthType::IndependentSuccess,
        ];
        for t in types {
            let mut sov = SovereigntyLevel::new();
            let before = sov.level;
            sov.record_growth(t, "test".into(), 0);
            assert!(sov.level > before, "Every growth type must increase level");
        }
    }

    // -----------------------------------------------------------------------
    // Guardian mode: suggestions have explanation + "no thanks"
    // -----------------------------------------------------------------------

    #[test]
    fn guardian_frustration_has_explanation_and_decline() {
        let state = CognitiveState {
            valence: -0.8,
            arousal: 0.9,
            stress: 0.3,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "fractions", 400, 0.1, 5, 5, &[]);
        let suggestion = result.suggestion.expect("Guardian should suggest");
        assert!(!suggestion.is_safety_concern);
        assert!(suggestion.message.contains("tricky"));
        // Must have a decline option
        assert!(
            suggestion.options.iter().any(|o| o.is_decline),
            "Guardian suggestions must have a 'no thanks' option"
        );
    }

    // -----------------------------------------------------------------------
    // Guide mode: multiple choices presented
    // -----------------------------------------------------------------------

    #[test]
    fn guide_frustration_offers_multiple_choices() {
        let state = CognitiveState {
            valence: -0.8,
            arousal: 0.9,
            stress: 0.3,
            ..default_state()
        };
        let result = adapt(&state, &guide(), "fractions", 400, 0.1, 5, 5, &[]);
        let suggestion = result.suggestion.expect("Guide should suggest");
        assert!(
            suggestion.options.len() >= 3,
            "Guide mode should offer multiple choices, got {}",
            suggestion.options.len()
        );
        assert!(suggestion.options.iter().any(|o| o.is_decline));
    }

    // -----------------------------------------------------------------------
    // Mirror mode: no unsolicited suggestions
    // -----------------------------------------------------------------------

    #[test]
    fn mirror_mode_no_suggestions() {
        let state = CognitiveState {
            valence: -0.8,
            arousal: 0.9,
            stress: 0.3,
            ..default_state()
        };
        let result = adapt(&state, &mirror(), "fractions", 400, 0.1, 5, 5, &[]);
        assert!(
            result.suggestion.is_none(),
            "Mirror mode should not make unsolicited suggestions"
        );
    }

    // -----------------------------------------------------------------------
    // Autonomous mode: no unsolicited suggestions
    // -----------------------------------------------------------------------

    #[test]
    fn autonomous_mode_no_suggestions() {
        let state = default_state();
        let result = adapt(&state, &autonomous(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.suggestion.is_none());
    }

    // -----------------------------------------------------------------------
    // Safety threshold
    // -----------------------------------------------------------------------

    #[test]
    fn safety_threshold_explains_who_set_it() {
        let state = CognitiveState {
            stress: 0.9, // above default threshold 0.85
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        let suggestion = result.suggestion.expect("Safety should trigger");
        assert!(suggestion.is_safety_concern);
        assert!(suggestion.message.contains("teacher"));
    }

    #[test]
    fn safety_threshold_has_im_ok_option() {
        let state = CognitiveState {
            stress: 0.9,
            ..default_state()
        };
        let result = adapt(&state, &guide(), "math", 500, 0.8, 0, 5, &[]);
        let suggestion = result.suggestion.expect("Safety should trigger");
        assert!(
            suggestion.options.iter().any(|o| o.is_decline),
            "Safety concern must have an 'I'm OK' option"
        );
    }

    #[test]
    fn safety_threshold_is_configurable() {
        let state = CognitiveState {
            stress: 0.7,
            ..default_state()
        };
        // With default threshold (0.85), this should NOT trigger.
        let no_trigger = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(
            no_trigger.suggestion.is_none()
                || !no_trigger
                    .suggestion
                    .as_ref()
                    .map_or(false, |s| s.is_safety_concern),
            "Default threshold should not trigger at stress 0.7"
        );
        // With custom threshold (0.6), it SHOULD trigger.
        let trigger = adapt_content(
            &state,
            &guardian(),
            "math",
            500,
            0.8,
            0,
            5,
            &[],
            0.6, // lower threshold
        );
        let suggestion = trigger.suggestion.expect("Custom threshold should trigger");
        assert!(suggestion.is_safety_concern);
    }

    #[test]
    fn safety_triggers_in_all_modes() {
        let state = CognitiveState {
            stress: 0.9,
            ..default_state()
        };
        for sov in [guardian(), guide(), mirror(), autonomous()] {
            let result = adapt(&state, &sov, "math", 500, 0.8, 0, 5, &[]);
            assert!(
                result
                    .suggestion
                    .as_ref()
                    .map_or(false, |s| s.is_safety_concern),
                "Safety should trigger in {:?} mode",
                sov.mode()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Metacognitive prompts
    // -----------------------------------------------------------------------

    #[test]
    fn metacognitive_guardian_streak() {
        let prompt =
            metacognitive_prompt(&guardian(), true, 3, "addition");
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("3 in a row"));
    }

    #[test]
    fn metacognitive_guardian_failure() {
        let prompt = metacognitive_prompt(&guardian(), false, 0, "math");
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("Mistakes help"));
    }

    #[test]
    fn metacognitive_guide_success() {
        let prompt = metacognitive_prompt(&guide(), true, 2, "math");
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("How did you figure"));
    }

    #[test]
    fn metacognitive_guide_long_streak() {
        let prompt = metacognitive_prompt(&guide(), true, 5, "math");
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("too easy"));
    }

    #[test]
    fn metacognitive_mirror_no_prompt() {
        assert!(metacognitive_prompt(&mirror(), true, 10, "math").is_none());
    }

    #[test]
    fn metacognitive_autonomous_no_prompt() {
        assert!(metacognitive_prompt(&autonomous(), false, 0, "math").is_none());
    }

    // -----------------------------------------------------------------------
    // Rewrite permission respects sovereignty
    // -----------------------------------------------------------------------

    #[test]
    fn rewrite_guardian_applies_automatically() {
        let result = suggest_rewrite(
            &guardian(),
            "Sam purchased 3 items.",
            &TextComplexity::Simplified,
            3,
        );
        assert!(matches!(result, RewriteResult::Applied { .. }));
        if let RewriteResult::Applied { rewritten, .. } = result {
            assert!(rewritten.contains("bought"));
        }
    }

    #[test]
    fn rewrite_guide_offers_choice() {
        let result = suggest_rewrite(
            &guide(),
            "Sam purchased 3 items.",
            &TextComplexity::Simplified,
            3,
        );
        assert!(matches!(result, RewriteResult::Offered { .. }));
    }

    #[test]
    fn rewrite_mirror_shows_original() {
        let result = suggest_rewrite(
            &mirror(),
            "Sam purchased 3 items.",
            &TextComplexity::Simplified,
            3,
        );
        assert!(matches!(result, RewriteResult::Available { .. }));
    }

    // -----------------------------------------------------------------------
    // Peer suggestion is an offer, not assignment
    // -----------------------------------------------------------------------

    #[test]
    fn peer_suggestion_in_guardian_mode() {
        let state = CognitiveState {
            social_readiness: 0.7,
            valence: 0.1,
            arousal: 0.3,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "geometry", 300, 0.2, 4, 5, &[]);
        assert!(result.peer_suggestion.is_some());
        let peer = result.peer_suggestion.unwrap();
        assert!(peer.peer_criteria.needs_patience);
    }

    #[test]
    fn no_peer_suggestion_in_mirror_mode() {
        let state = CognitiveState {
            social_readiness: 0.7,
            valence: 0.1,
            arousal: 0.3,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt(&state, &mirror(), "geometry", 300, 0.2, 4, 5, &[]);
        assert!(
            result.peer_suggestion.is_none(),
            "Mirror mode should not suggest peers"
        );
    }

    // -----------------------------------------------------------------------
    // Student decline is recorded without penalty
    // -----------------------------------------------------------------------

    #[test]
    fn decline_option_always_present_in_suggestions() {
        // Test across different suggestion triggers
        let frustrated = CognitiveState {
            valence: -0.8,
            arousal: 0.9,
            stress: 0.3,
            ..default_state()
        };
        let result = adapt(&frustrated, &guardian(), "math", 400, 0.1, 5, 5, &[]);
        if let Some(s) = &result.suggestion {
            assert!(s.options.iter().any(|o| o.is_decline));
        }

        // Challenge suggestion
        let excelling = CognitiveState {
            phi: 0.7,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt(&excelling, &guardian(), "math", 900, 0.9, 0, 5, &[]);
        if let Some(s) = &result.suggestion {
            assert!(s.options.iter().any(|o| o.is_decline));
        }
    }

    // -----------------------------------------------------------------------
    // Text simplification (preserved utility tests)
    // -----------------------------------------------------------------------

    #[test]
    fn simplify_replaces_complex_words() {
        let text = "Sam purchased 5 bags. How many remaining?";
        let simplified = simplify_text(text, 3);
        assert!(simplified.contains("bought"));
        assert!(simplified.contains("left"));
    }

    #[test]
    fn simplify_grade_dependent_rectangle() {
        let early = simplify_text("Draw a rectangle.", 4);
        assert!(early.contains("shape"));
        let later = simplify_text("Draw a rectangle.", 6);
        assert!(later.contains("rectangle"));
    }

    #[test]
    fn simplify_grade_dependent_multiplication() {
        let early = simplify_text("Use multiplication.", 3);
        assert!(early.contains("times"));
        let later = simplify_text("Use multiplication.", 5);
        assert!(later.contains("multiplication"));
    }

    // -----------------------------------------------------------------------
    // Personalization (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn personalize_dinosaurs() {
        let result = personalize_text("Sam has 3 bags with 4 apples each.", "dinosaurs");
        assert!(result.contains("nests"));
        assert!(result.contains("dinosaur eggs"));
    }

    #[test]
    fn personalize_minecraft() {
        let result = personalize_text("There are 5 boxes of stickers.", "minecraft");
        assert!(result.contains("chests"));
        assert!(result.contains("diamonds"));
    }

    #[test]
    fn personalize_unknown_uses_raw() {
        let result = personalize_text("There are 3 groups of items.", "robots");
        assert!(result.contains("groups"));
        assert!(result.contains("robots"));
    }

    // -----------------------------------------------------------------------
    // Equation extraction (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn extract_equation_from_word_problem() {
        let eq = extract_equation("Sam has 3 + 4 apples. How many?");
        assert!(eq.contains("3 + 4"));
        assert!(!eq.contains("Sam"));
    }

    #[test]
    fn extract_equation_no_operators_takes_first_sentence() {
        let eq = extract_equation("Sam has some apples. He gave them away.");
        assert_eq!(eq, "Sam has some apples");
    }

    // -----------------------------------------------------------------------
    // Grounding exercises (now suggestions, not impositions)
    // -----------------------------------------------------------------------

    #[test]
    fn young_student_high_stress_gets_sensory_grounding() {
        let suggestion = grounding_exercise(3, 0.85);
        assert!(suggestion.message.contains("5-4-3-2-1"));
        assert!(suggestion.options.iter().any(|o| o.is_decline));
    }

    #[test]
    fn young_student_moderate_stress_gets_breathing() {
        let suggestion = grounding_exercise(2, 0.5);
        assert!(suggestion.message.contains("breaths"));
        assert!(suggestion.options.iter().any(|o| o.is_decline));
    }

    #[test]
    fn older_student_gets_reappraisal() {
        let suggestion = grounding_exercise(8, 0.85);
        assert!(suggestion.message.contains("challenging"));
        assert_eq!(
            suggestion.suggestion_type,
            SuggestionType::TryDifferentApproach
        );
    }

    // -----------------------------------------------------------------------
    // Signal derivation (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn derive_cognitive_state_cortisol() {
        let state =
            derive_cognitive_state(0.5, 0.5, 0.5, 0.2, 0.8, MotorCommand::NoOp);
        assert!((state.stress - 0.64).abs() < 0.01);
    }

    #[test]
    fn derive_cognitive_state_valence() {
        let state =
            derive_cognitive_state(0.5, 0.5, 0.3, 0.2, 0.8, MotorCommand::NoOp);
        assert!((state.valence - (-0.34)).abs() < 0.01);
    }

    #[test]
    fn derive_cognitive_state_permeability_sigmoid() {
        let open =
            derive_cognitive_state(0.5, 0.5, 0.5, 0.8, 0.1, MotorCommand::NoOp);
        assert!(open.permeability > 0.7);

        let closed =
            derive_cognitive_state(0.5, 0.5, 0.3, 0.1, 0.9, MotorCommand::NoOp);
        assert!(closed.permeability < 0.4);
    }

    #[test]
    fn derive_cognitive_state_oxytocin() {
        let state =
            derive_cognitive_state(0.5, 0.5, 0.6, 0.8, 0.3, MotorCommand::NoOp);
        assert!((state.social_readiness - 0.72).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // FEP motor command -> difficulty (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn exploration_trigger_increases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::ExplorationTrigger,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta > 0.0);
    }

    #[test]
    fn memory_consolidate_decreases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::MemoryConsolidate,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta < 0.0);
    }

    #[test]
    fn learning_rate_adjust_decreases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::LearningRateAdjust,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta < 0.0);
    }

    #[test]
    fn reflection_initiate_zero_difficulty_change() {
        let state = CognitiveState {
            motor_command: MotorCommand::ReflectionInitiate,
            permeability: 1.0,
            stress: 0.2,
            ..default_state()
        };
        // Use mirror mode so the reflection suggestion doesn't fire,
        // keeping the test focused on difficulty.
        let result = adapt(&state, &mirror(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta.abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Free energy -> difficulty (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn noop_high_free_energy_decreases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::NoOp,
            free_energy: 0.8,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.5, 0, 5, &[]);
        assert!(result.difficulty_delta < 0.0);
    }

    #[test]
    fn noop_low_free_energy_increases_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::NoOp,
            free_energy: 0.2,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta > 0.0);
    }

    #[test]
    fn noop_moderate_free_energy_no_change() {
        let state = CognitiveState {
            motor_command: MotorCommand::NoOp,
            free_energy: 0.5,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta.abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Permeability scaling (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn low_permeability_narrows_difficulty() {
        let state = CognitiveState {
            motor_command: MotorCommand::ExplorationTrigger,
            permeability: 0.2,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.difficulty_delta < 0.1);
        assert!(result.difficulty_delta > 0.0);
    }

    #[test]
    fn high_permeability_full_difficulty_range() {
        let state = CognitiveState {
            motor_command: MotorCommand::ExplorationTrigger,
            permeability: 1.0,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!((result.difficulty_delta - 0.3).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Modality selection (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn high_focus_selects_text_modality() {
        let state = CognitiveState { focus: 0.8, ..default_state() };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::Text);
    }

    #[test]
    fn high_social_readiness_selects_kinesthetic() {
        let state = CognitiveState {
            focus: 0.5,
            social_readiness: 0.7,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::Kinesthetic);
    }

    #[test]
    fn low_arousal_selects_visual() {
        let state = CognitiveState {
            focus: 0.4,
            social_readiness: 0.3,
            arousal: 0.2,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
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
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.modality, Modality::MultiModal);
    }

    // -----------------------------------------------------------------------
    // Text complexity selection (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn failures_with_interests_triggers_personalized() {
        let interests = vec!["dinosaurs".to_string()];
        let state = CognitiveState { focus: 0.5, ..default_state() };
        let result =
            adapt(&state, &guardian(), "multiplication", 300, 0.2, 3, 5, &interests);
        assert!(matches!(
            result.text_complexity,
            TextComplexity::Personalized { .. }
        ));
    }

    #[test]
    fn failures_no_interests_triggers_simplified() {
        let state = CognitiveState { focus: 0.5, ..default_state() };
        let result = adapt(&state, &guardian(), "multiplication", 300, 0.2, 3, 5, &[]);
        assert_eq!(result.text_complexity, TextComplexity::Simplified);
    }

    #[test]
    fn low_focus_triggers_minimal() {
        let state = CognitiveState { focus: 0.2, ..default_state() };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.text_complexity, TextComplexity::Minimal);
    }

    #[test]
    fn normal_state_keeps_standard() {
        let state = CognitiveState { focus: 0.5, ..default_state() };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert_eq!(result.text_complexity, TextComplexity::Standard);
    }

    // -----------------------------------------------------------------------
    // Challenge / celebration (now sovereignty-aware)
    // -----------------------------------------------------------------------

    #[test]
    fn high_phi_high_mastery_offers_challenge() {
        let state = CognitiveState {
            phi: 0.7,
            stress: 0.2,
            motor_command: MotorCommand::NoOp,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "algebra", 900, 0.9, 0, 8, &[]);
        let suggestion = result.suggestion.expect("Should celebrate progress");
        assert_eq!(suggestion.suggestion_type, SuggestionType::CelebrateProgress);
        assert!(suggestion.options.iter().any(|o| o.is_decline));
    }

    #[test]
    fn reflection_motor_offers_reflection() {
        let state = CognitiveState {
            motor_command: MotorCommand::ReflectionInitiate,
            stress: 0.2,
            ..default_state()
        };
        let result = adapt(&state, &guardian(), "math", 500, 0.8, 0, 5, &[]);
        let suggestion = result.suggestion.expect("Should offer reflection");
        assert_eq!(suggestion.suggestion_type, SuggestionType::Reflect);
    }

    // -----------------------------------------------------------------------
    // Reasoning trace (preserved)
    // -----------------------------------------------------------------------

    #[test]
    fn reasoning_trace_includes_all_signals() {
        let result = adapt(&default_state(), &guardian(), "math", 500, 0.8, 0, 5, &[]);
        assert!(result.reasoning.contains("Phi="));
        assert!(result.reasoning.contains("FE="));
        assert!(result.reasoning.contains("valence="));
        assert!(result.reasoning.contains("Mode:"));
    }

    // -----------------------------------------------------------------------
    // Edge cases (preserved)
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
        let result = adapt(&state, &guardian(), "math", 0, 0.0, 0, 0, &[]);
        assert!(result.difficulty_delta.abs() < f32::EPSILON);
    }

    #[test]
    fn all_max_state_triggers_safety() {
        let state = CognitiveState {
            phi: 1.0,
            free_energy: 1.5,
            valence: 1.0,
            arousal: 1.0,
            focus: 1.0,
            motivation: 1.0,
            stress: 1.0, // above safety threshold
            social_readiness: 1.0,
            permeability: 1.0,
            motor_command: MotorCommand::ExplorationTrigger,
        };
        let result = adapt(&state, &guardian(), "math", 1000, 1.0, 100, 13, &[]);
        let suggestion = result.suggestion.expect("Safety should trigger");
        assert!(suggestion.is_safety_concern);
    }

    #[test]
    fn serialization_roundtrip() {
        let result = adapt(&default_state(), &guardian(), "math", 500, 0.8, 0, 5, &[]);
        let json = serde_json::to_string(&result).expect("serialize");
        let _: ContentAdaptation =
            serde_json::from_str(&json).expect("deserialize");
    }

    #[test]
    fn sovereignty_serialization_roundtrip() {
        let mut sov = SovereigntyLevel::new();
        sov.record_growth(
            SovereigntyGrowthType::AskedForHelp,
            "test".into(),
            42,
        );
        let json = serde_json::to_string(&sov).expect("serialize");
        let deser: SovereigntyLevel =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.level, sov.level);
        assert_eq!(deser.growth_events.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Consecutive success -> sovereignty growth opportunity
    // -----------------------------------------------------------------------

    #[test]
    fn consecutive_success_metacognitive_prompt_exists() {
        // After 5 correct in Guide mode, the system should prompt about difficulty.
        let prompt = metacognitive_prompt(&guide(), true, 5, "fractions");
        assert!(prompt.is_some());
        let text = prompt.unwrap();
        assert!(text.contains("too easy"));
    }

    // -----------------------------------------------------------------------
    // Rewrite integration (preserved + sovereignty)
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
}
