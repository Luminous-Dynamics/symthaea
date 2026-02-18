//! Public types returned by the cognitive loop.
//!
//! Extracted from `mod.rs` to reduce file size while preserving all public APIs.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE URGENCY — adaptive subsystem scheduling
// ═══════════════════════════════════════════════════════════════════════════════

/// Urgency level controlling how many subsystems run each cycle.
///
/// Instead of fixed "every Nth cycle" throttling, urgency adapts to the
/// system's current needs:
/// - **Critical**: High error or surprise — run everything for maximum adaptation
/// - **Normal**: Standard processing — run most subsystems
/// - **Cruise**: Low error, stable state — skip expensive subsystems to save compute
///
/// Subsystems decide per-urgency whether to run:
/// - Core pipeline (HDC→CfC→predict→learn): always runs
/// - Moral evaluation: Critical+Normal (skip in Cruise unless new input)
/// - Enhanced FEP: Critical always, Normal every 4th, Cruise every 8th
/// - Stability regime: Critical+Normal, Cruise every 4th
/// - Consciousness monitors (resonance, quantum, temporal): Normal+Critical only
/// - Master equation: Critical every 5th, Normal every 10th, Cruise every 20th
/// - Body awareness (virtual body, affective, embodied): Normal+Critical, Cruise every 2nd
/// - Self models (meta-cognition, narrative, predictive mind/self): C=1, N=2, Cr=4
/// - Workspace (attention schema, GWT, cross-modal, narrative-GWT): C=1, N=2, Cr=4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CycleUrgency {
    /// High prediction error or surprise — run all subsystems
    Critical,
    /// Standard processing
    #[default]
    Normal,
    /// Low error, stable state — minimal subsystem overhead
    Cruise,
}

impl CycleUrgency {
    /// Compute urgency from current cycle state.
    ///
    /// - `prediction_error`: current cycle's prediction error
    /// - `learning_threshold`: config threshold for "significant" error
    /// - `surprise_triggered`: whether the surprise bridge triggered this cycle
    /// - `consecutive_low_error`: how many consecutive cycles have had error < threshold
    pub fn from_state(
        prediction_error: f32,
        learning_threshold: f32,
        surprise_triggered: bool,
        consecutive_low_error: u32,
    ) -> Self {
        if surprise_triggered || prediction_error > learning_threshold * 3.0 {
            CycleUrgency::Critical
        } else if prediction_error > learning_threshold || consecutive_low_error < 10 {
            CycleUrgency::Normal
        } else {
            CycleUrgency::Cruise
        }
    }

    /// Whether this urgency level should run a subsystem at the given cycle interval.
    /// Returns true if the subsystem should run this cycle.
    #[inline]
    pub fn should_run(&self, cycle: usize, critical_interval: usize, normal_interval: usize, cruise_interval: usize) -> bool {
        let interval = match self {
            CycleUrgency::Critical => critical_interval,
            CycleUrgency::Normal => normal_interval,
            CycleUrgency::Cruise => cruise_interval,
        };
        interval == 0 || cycle % interval == 0
    }

    /// Whether to run expensive consciousness monitors (resonance, quantum, temporal).
    #[inline]
    pub fn run_consciousness_monitors(&self) -> bool {
        matches!(self, CycleUrgency::Critical | CycleUrgency::Normal)
    }
}

/// Metadata about internal decision-making during a cycle.
///
/// Provides observability into which subsystems influenced the cycle's output,
/// enabling debugging of "why did the agent do that?" questions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleMetadata {
    /// Whether the surprise exploration bridge triggered exploration this cycle
    pub surprise_triggered: bool,

    /// Whether the prefrontal cortex vetoed or modified the response
    pub prefrontal_veto: bool,

    /// Confidence score from the reasoning engine (0.0 = unused/off, >0 = active)
    pub reasoning_confidence: f32,

    /// Description of the exploration action taken (if any)
    pub exploration_action: Option<String>,

    /// Whether the reasoning engine's tool gate blocked an action this cycle.
    /// When true, the system used a fallback strategy instead.
    pub reasoning_gate_blocked: bool,

    /// Fallback strategy selected when gating blocked an action (if any)
    pub reasoning_fallback: Option<String>,

    /// Best action from MCTS planning (Tier 1+), if planning ran
    pub reasoning_plan_action: Option<usize>,

    /// MCTS plan confidence (0.0 = no plan, >0 = plan confidence)
    pub reasoning_plan_confidence: f32,

    /// Human-readable reasoning narrative (Tier 2, best-effort)
    pub reasoning_narrative: Option<String>,

    /// Meta-cognitive self-model accuracy (0.0 = uncertain, 1.0 = perfect self-knowledge)
    pub meta_cognitive_accuracy: f32,

    /// Meta-cognitive recursion depth (0 = off, 1 = basic, 2+ = recursive self-modeling)
    pub meta_cognitive_depth: u8,

    /// Narrative self-model's integrated information (0.0 = off/no self, >0 = active self-Φ)
    pub narrative_self_phi: f64,

    /// Virtual body phi modulation (1.0 = neutral, >1 = body boosts consciousness)
    pub body_phi_modulation: f64,

    /// Virtual body affect valence (-1 to 1)
    pub body_valence: f32,

    /// Virtual body affect arousal (0 to 1)
    pub body_arousal: f32,

    /// Master Consciousness Equation level (0.0 to 1.0).
    /// Comprehensive consciousness metric combining Phi, broadcast, working memory,
    /// attention, recurrence, embodiment, knowledge, narrative, and social factors.
    /// Updated every 10th cycle; 0.0 when not yet computed.
    pub consciousness_level: f64,

    /// Predictive self-model safety score (1.0 = safe, 0.0 = unsafe).
    /// 0.0 when predictive self is not enabled.
    pub predictive_self_safety: f32,

    /// Attention schema focus intensity (0.0 to 1.0).
    /// 0.0 when attention schema is not enabled.
    pub attention_schema_focus: f32,

    /// Whether a GWT broadcast occurred this cycle.
    pub gwt_broadcast: bool,

    /// Consciousness resonance dominant frequency (Hz).
    /// 0.0 when resonance is not enabled or no history.
    pub resonance_frequency: f64,

    /// Quantum coherence level (0.0 to 1.0).
    /// 0.0 when quantum coherence is not enabled.
    pub quantum_coherence_level: f64,

    /// Temporal consciousness coherence (0.0 to 1.0).
    /// 0.0 when temporal consciousness is not enabled.
    pub temporal_coherence_score: f64,

    /// Whether temporal consciousness analysis detected a discontinuity.
    pub temporal_discontinuity: bool,

    /// Embodied cognition phi modulation (1.0 = neutral).
    /// 1.0 when embodied cognition is not enabled.
    pub embodied_phi_modulation: f64,

    /// Embodied cognition agency score (0.0 to 1.0).
    /// 0.0 when embodied cognition is not enabled.
    pub embodied_agency: f64,

    /// Whether the narrative-GWT integration vetoed this cycle's action.
    pub narrative_gwt_veto: bool,

    /// Self-Phi from the narrative-GWT integration (0.0 = off/not enabled).
    pub narrative_gwt_self_phi: f64,

    /// Unified Living Mind vitality (0.0 to 1.0).
    /// Measures overall "aliveness" of the system via life-mind continuity.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_vitality: f64,

    /// Unified Living Mind coherence (0.0 to 1.0).
    /// Measures integration quality of autopoietic, enactive, and predictive subsystems.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_coherence: f64,

    /// Cycle urgency level (Critical/Normal/Cruise).
    /// Determines how many subsystems ran this cycle.
    pub urgency: CycleUrgency,

    /// Number of insights gained from dream replay this cycle (0 = no dreaming).
    pub dream_insights: usize,

    /// Best Phi improvement discovered by dream counterfactuals (0.0 = no improvement).
    pub dream_phi_improvement: f32,

    /// Total accumulated wisdom entries from dreaming.
    pub dream_wisdom_count: usize,

    /// Predictive processing free energy (0.0 when off).
    pub predictive_free_energy: f64,

    /// Predictive processing phi modulation (1.0 when off — neutral).
    pub predictive_phi_modulation: f64,

    /// Cross-modal binding strength (0.0 when off).
    pub cross_modal_binding_strength: f32,

    /// Cross-modal integration Phi (0.0 when off).
    pub cross_modal_phi: f64,

    /// Affective bridge valence (-1 to 1, 0.0 when off).
    pub affective_valence: f32,

    /// Affective bridge arousal (0 to 1, 0.5 when off — neutral).
    pub affective_arousal: f32,

    /// Per-module timing (microseconds). 0 = module disabled or not run this cycle.
    pub module_timings_us: ModuleTimings,
}

/// Per-module execution timings in microseconds for overhead profiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleTimings {
    pub affective_bridge: u64,
    pub predictive_processing: u64,
    pub cross_modal_binding: u64,
    pub surprise_exploration: u64,
    pub prefrontal: u64,
    pub meta_cognition: u64,
    pub narrative_self: u64,
    pub gwt: u64,
    pub virtual_body: u64,
    pub embodied_cognition: u64,
    pub dream_replay: u64,
    pub moral_algebra: u64,
    pub consciousness_resonance: u64,
    pub temporal_consciousness: u64,
    pub attention_schema: u64,
    pub narrative_gwt: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ATTESTATION RECORD — for governance bridge consumption
// ═══════════════════════════════════════════════════════════════════════════════

/// Record of a Phi measurement from a cognitive cycle, ready for attestation.
///
/// The cognitive loop produces these after each cycle where Phi is computed.
/// The personal cluster (or symthaea-mycelix-bridge) can consume these records,
/// sign them with the agent's cryptographic key, and submit to governance as
/// authenticated `PhiAttestation` entries.
///
/// This is a lightweight struct with no external dependencies — the bridge crate
/// converts it to the Holochain-compatible `PhiAttestationData` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiAttestationRecord {
    /// Unified Phi value from this cycle (clamped to [0.0, 1.0])
    pub phi: f64,

    /// Cognitive cycle number that produced this Phi
    pub cycle_id: u64,

    /// Timestamp in microseconds since UNIX epoch
    pub captured_at_us: u64,

    /// Prediction error at time of measurement (context for Phi quality)
    pub prediction_error: f32,

    /// Urgency level during measurement (Critical/Normal/Cruise)
    pub urgency: CycleUrgency,
}

impl PhiAttestationRecord {
    /// Canonical message for signing: deterministic byte representation.
    /// The bridge crate signs this with the agent's Ed25519 key.
    pub fn sign_message(&self, agent_did: &str) -> Vec<u8> {
        format!(
            "symthaea-phi-attestation:v1:{}:{:.6}:{}:{}",
            agent_did, self.phi, self.cycle_id, self.captured_at_us,
        )
        .into_bytes()
    }
}

/// Result of a single cognitive cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// LTC output (interpretation of current state)
    pub output: Vec<f32>,

    /// Prediction error for this cycle
    pub prediction_error: f32,

    /// Peak attention value from encoder
    pub peak_attention: f32,

    /// Detected primitives in input
    pub detected_primitives: Vec<String>,

    /// Whether learning occurred this cycle
    pub learning_occurred: bool,

    /// Training loss (if learning occurred)
    pub training_loss: Option<f32>,

    /// Cycle timing (microseconds)
    pub cycle_time_us: u64,

    /// Internal decision-making metadata for observability
    pub metadata: CycleMetadata,

    /// Signed output for identity verification (when identity feature enabled)
    /// Contains Ed25519 signature over output hash and agent metadata
    #[cfg(feature = "identity")]
    pub signed_output: Option<crate::identity::SignedOutput>,

    /// Agent assurance level at time of processing (when identity feature enabled)
    #[cfg(feature = "identity")]
    pub assurance_level: crate::identity::AssuranceLevel,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL JUDGMENT SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

/// Summary of moral evaluation for an action or input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralJudgmentSummary {
    /// The input text that was evaluated
    pub input: String,

    /// Overall moral verdict (Good, Bad, Neutral, ConsentViolation)
    pub verdict: String,

    /// Deontological verdict (Permissible, Impermissible, Obligatory, Supererogatory)
    pub deontological_verdict: String,

    /// List of detected moral violations (e.g., "dishonesty", "theft")
    pub violations: Vec<String>,

    /// List of moral satisfactions (e.g., "honesty", "beneficence")
    pub satisfactions: Vec<String>,

    /// Whether consent was violated
    pub consent_violation: bool,

    /// Moral score (-1.0 to 1.0, negative = bad, positive = good)
    pub moral_score: f32,

    /// Confidence in the moral judgment (0.0 to 1.0)
    pub confidence: f32,
}

impl Default for MoralJudgmentSummary {
    fn default() -> Self {
        Self {
            input: String::new(),
            verdict: "Neutral".to_string(),
            deontological_verdict: "Permissible".to_string(),
            violations: Vec::new(),
            satisfactions: Vec::new(),
            consent_violation: false,
            moral_score: 0.0,
            confidence: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive behavior parameters derived from consciousness state
///
/// These parameters allow the system to self-regulate based on its
/// current consciousness pattern, creating a truly self-aware loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBehavior {
    /// Learning rate multiplier (0.1 to 2.0)
    /// Higher when focused/confident, lower when uncertain
    pub learning_rate_multiplier: f32,

    /// Speech rate multiplier (0.7 to 1.3)
    /// Slower when contemplative/uncertain, faster when excited/focused
    pub speech_rate_multiplier: f32,

    /// Pause duration multiplier (0.5 to 2.0)
    /// Longer pauses when contemplative, shorter when excited
    pub pause_multiplier: f32,

    /// Attention sensitivity (0.5 to 1.5)
    /// Higher sensitivity when exploratory, lower when focused
    pub attention_sensitivity: f32,

    /// Exploration factor (0.0 to 1.0)
    /// Higher when exploratory/uncertain, lower when focused
    pub exploration_factor: f32,

    /// Confidence level (0.0 to 1.0)
    /// Derived from pattern + coherence + voice quality
    pub confidence: f32,

    /// Should pause learning (e.g., during transitions)
    pub pause_learning: bool,

    /// Recommended action hint
    pub action_hint: ActionHint,
}

/// Recommended action based on consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionHint {
    /// Continue normally
    Continue,
    /// Slow down, be more deliberate
    SlowDown,
    /// Speed up, more confident
    SpeedUp,
    /// Pause and stabilize
    Stabilize,
    /// Explore alternatives
    Explore,
    /// Seek clarification or more input
    SeekInput,
}

impl Default for AdaptiveBehavior {
    fn default() -> Self {
        Self {
            learning_rate_multiplier: 1.0,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            attention_sensitivity: 1.0,
            exploration_factor: 0.3,
            confidence: 0.5,
            pause_learning: false,
            action_hint: ActionHint::Continue,
        }
    }
}

impl AdaptiveBehavior {
    /// Compute adaptive behavior from consciousness pattern and metrics
    pub fn from_consciousness_state(
        pattern: crate::dynamics::temporal_signatures::ConsciousnessPattern,
        pattern_confidence: f32,
        coherence: f32,
        voice_confidence: f32,
    ) -> Self {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;

        // Base confidence from all sources
        let confidence =
            (pattern_confidence * 0.4 + coherence * 0.3 + voice_confidence * 0.3).clamp(0.0, 1.0);

        match pattern {
            ConsciousnessPattern::Focused => Self {
                learning_rate_multiplier: 1.3 + confidence * 0.4, // 1.3 to 1.7
                speech_rate_multiplier: 1.05 + confidence * 0.15, // 1.05 to 1.2
                pause_multiplier: 0.7,
                attention_sensitivity: 0.7, // Less distracted
                exploration_factor: 0.1,    // Stay on track
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SpeedUp,
            },

            ConsciousnessPattern::Contemplative => Self {
                learning_rate_multiplier: 0.8,
                speech_rate_multiplier: 0.85,
                pause_multiplier: 1.5, // Longer pauses for reflection
                attention_sensitivity: 1.0,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SlowDown,
            },

            ConsciousnessPattern::Excited => Self {
                learning_rate_multiplier: 1.1,
                speech_rate_multiplier: 1.15,
                pause_multiplier: 0.6,      // Quick transitions
                attention_sensitivity: 1.3, // More reactive
                exploration_factor: 0.4,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Exploratory => Self {
                learning_rate_multiplier: 1.0,
                speech_rate_multiplier: 0.95,
                pause_multiplier: 1.0,
                attention_sensitivity: 1.4, // High sensitivity to new info
                exploration_factor: 0.7,    // Actively explore
                confidence: confidence * 0.8,
                pause_learning: false,
                action_hint: ActionHint::Explore,
            },

            ConsciousnessPattern::Resting => Self {
                learning_rate_multiplier: 0.6,
                speech_rate_multiplier: 0.9,
                pause_multiplier: 1.2,
                attention_sensitivity: 0.8,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Transitioning => Self {
                learning_rate_multiplier: 0.3, // Minimal learning during transition
                speech_rate_multiplier: 0.8,
                pause_multiplier: 1.8, // Pause to stabilize
                attention_sensitivity: 1.0,
                exploration_factor: 0.3,
                confidence: confidence * 0.5,
                pause_learning: true, // Pause learning
                action_hint: ActionHint::Stabilize,
            },

            ConsciousnessPattern::Uncertain => Self {
                learning_rate_multiplier: 0.4, // Careful learning
                speech_rate_multiplier: 0.75,  // Slow down
                pause_multiplier: 2.0,         // Long pauses
                attention_sensitivity: 1.2,
                exploration_factor: 0.5,
                confidence: confidence * 0.3,
                pause_learning: false,
                action_hint: ActionHint::SeekInput,
            },
        }
    }

    /// Get effective learning rate with all modulations
    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        if self.pause_learning {
            0.0
        } else {
            base_rate * self.learning_rate_multiplier
        }
    }

    /// Check if the system should seek more input/clarification
    pub fn should_seek_input(&self) -> bool {
        self.action_hint == ActionHint::SeekInput || self.confidence < 0.3
    }

    /// Check if the system is in a confident state
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.6 && !self.pause_learning
    }

    /// Get a human-readable description of current state
    pub fn description(&self) -> &'static str {
        match self.action_hint {
            ActionHint::Continue => "Operating normally",
            ActionHint::SlowDown => "Deliberating carefully",
            ActionHint::SpeedUp => "Confident and focused",
            ActionHint::Stabilize => "Stabilizing state",
            ActionHint::Explore => "Exploring possibilities",
            ActionHint::SeekInput => "Seeking clarification",
        }
    }
}
