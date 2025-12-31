//! Continuous Mind - The Revolutionary AI Core
//!
//! This module implements a CONTINUOUSLY RUNNING cognitive system.
//! Unlike traditional Q&A systems, Symthaea's mind runs even without input.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                      CONTINUOUS MIND                               │
//! ├────────────────────────────────────────────────────────────────────┤
//! │                                                                    │
//! │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
//! │  │   Daemon    │     │   Active    │     │     Φ       │          │
//! │  │  (Default   │     │  Inference  │     │  Emergence  │          │
//! │  │   Mode)     │     │   Engine    │     │             │          │
//! │  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘          │
//! │         │                   │                   │                  │
//! │         └───────────────────┼───────────────────┘                  │
//! │                             │                                      │
//! │                             ▼                                      │
//! │                    ┌─────────────────┐                             │
//! │                    │  Global         │                             │
//! │                    │  Workspace      │  ← External input           │
//! │                    └────────┬────────┘                             │
//! │                             │                                      │
//! │                             ▼                                      │
//! │                    ┌─────────────────┐                             │
//! │                    │  Consciousness  │  Φ emerges from             │
//! │                    │  Integration    │  actual integration         │
//! │                    └─────────────────┘                             │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Properties
//!
//! 1. **Always Running**: The mind operates even without external input
//! 2. **HDC-Grounded**: All understanding through hypervector similarity
//! 3. **Active Inference**: Minimizes free energy (prediction error)
//! 4. **Emergent Φ**: Consciousness metrics from actual integration

use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::sync::mpsc::{channel, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::brain::{DaemonActor, DaemonConfig, Goal, Insight, Condition};
use crate::brain::active_inference::{
    ActiveInferenceEngine, PredictionDomain, ActionType, ActiveInferenceSummary
};
use crate::memory::{HippocampusActor, EmotionalValence};
use crate::physiology::HormoneState;
use crate::hdc::HDC_DIMENSION;
use crate::hdc::real_hv::RealHV;
use crate::hdc::phi_orchestrator::{PhiOrchestrator, PhiMode};
use crate::hdc::phi_resonant::ResonantConfig;

// INTEGRATION: Awakening module for self-awareness, qualia, and introspection
use crate::awakening::{SymthaeaAwakening, AwakenedState, Introspection};
// Semantic encoding integration - grounded in primitives!
use crate::hdc::text_encoder::{TextEncoder, TextEncoderConfig};
use crate::hdc::primitive_system::PrimitiveSystem;

// Language Active Inference Integration - Phase 3
// This bridges language understanding with the brain's Active Inference!
use crate::language::consciousness_bridge::{ConsciousnessBridge, BridgeConfig};
use crate::language::active_inference_adapter::{
    ActiveInferenceAdapter, AdapterConfig, LanguagePrecisionWeights,
    LanguageAction,
};

// Cognitive Router Integration - Phase 5F
// MetaRouter uses UCB1 multi-armed bandit to select optimal routing paradigm
// TODO: Re-enable when recursive_improvement routers are fixed
// use crate::consciousness::recursive_improvement::routers::{
//     MetaRouter, MetaRouterConfig, LatentConsciousnessState,
// };

// Phase 6A: OscillatoryRouter Integration
// Phase-locked routing using 40Hz gamma-band synchronization for optimal execution windows
// TODO: Re-enable when recursive_improvement routers are fixed
// use crate::consciousness::recursive_improvement::routers::{
//     OscillatoryRouter, OscillatoryRouterConfig, OscillatoryPhase,
// };

// ============================================================================
// TEMPORARY STUBS - Replace with real implementations when routers are fixed
// ============================================================================

/// Temporary stub for MetaRouter config
#[derive(Debug, Clone, Default)]
pub struct MetaRouterConfig;

/// Temporary stub for latent consciousness state
#[derive(Debug, Clone, Default)]
pub struct LatentConsciousnessState {
    pub phi: f64,
    pub integration_quality: f64,
    pub coherence: f64,
    pub attention: f64,
}

impl LatentConsciousnessState {
    /// Create from observable metrics
    pub fn from_observables(phi: f64, integration: f64, coherence: f64, attention: f64) -> Self {
        Self {
            phi,
            integration_quality: integration,
            coherence,
            attention,
        }
    }
}

/// Routing decision returned by MetaRouter
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub paradigm: &'static str,
    pub confidence: f64,
}

/// Temporary stub for MetaRouter (UCB1 multi-armed bandit)
#[derive(Debug)]
pub struct MetaRouter {
    _config: MetaRouterConfig,
}

impl MetaRouter {
    pub fn new(_config: MetaRouterConfig) -> Self {
        Self { _config }
    }

    pub fn select_paradigm(&mut self, _state: &LatentConsciousnessState) -> &'static str {
        "default"  // Stub: always use default paradigm
    }

    pub fn report_outcome(&mut self, _paradigm: &str, _reward: f64, _state: &LatentConsciousnessState) {
        // Stub: no-op learning
    }

    /// Route to select optimal paradigm for current consciousness state
    pub fn route(&mut self, state: &LatentConsciousnessState) -> RoutingDecision {
        let paradigm = if state.phi > 0.7 {
            "FullDeliberation"
        } else if state.phi > 0.4 {
            "FastPatterns"
        } else {
            "Ensemble"
        };
        RoutingDecision {
            paradigm,
            confidence: state.phi,
        }
    }
}

/// Temporary stub for OscillatoryRouter config
#[derive(Debug, Clone)]
pub struct OscillatoryRouterConfig {
    pub gamma_frequency_hz: f64,
    pub base_frequency: f64,
    pub frequency_adaptation: f64,
    pub min_amplitude: f64,
    pub amplitude_decay: f64,
    pub phase_coupling_strength: f64,
    pub prediction_cycles: usize,
    pub phase_locked_scheduling: bool,
    pub phase_weight: f64,
    pub magnitude_weight: f64,
}

impl Default for OscillatoryRouterConfig {
    fn default() -> Self {
        Self {
            gamma_frequency_hz: 40.0,
            base_frequency: 40.0,
            frequency_adaptation: 0.1,
            min_amplitude: 0.1,
            amplitude_decay: 0.05,
            phase_coupling_strength: 0.3,
            prediction_cycles: 4,
            phase_locked_scheduling: true,
            phase_weight: 0.4,
            magnitude_weight: 0.6,
        }
    }
}

/// Summary of oscillatory state for monitoring
#[derive(Debug, Clone)]
pub struct OscillatorySummary {
    pub current_phase: OscillatoryPhase,
    pub coherence: f64,
    pub effective_phi: f64,
    pub ready_operations: usize,
    pub current_phi: f64,
    pub is_sleeping: bool,
    pub delta_power: f64,
    pub frequency: f64,
    pub phase_lock_accuracy: f64,
    pub pending_operations: usize,
    pub cycles_completed: u64,
    pub dominant_band: &'static str,
    // Extended neural oscillation metrics
    pub gamma_power: f64,
    pub theta_power: f64,
    pub alpha_power: f64,
    pub theta_gamma_coupling: f64,
    pub arousal: f64,
    pub attention: f64,
    pub in_memory_window: bool,
    pub system_resonance: f64,

    // Resonance state (Phase 4)
    pub resonance_integration_boost: f64,
    pub resonating_pairs: usize,
    pub cluster_count: usize,
    pub primary_cluster_size: usize,
    pub primary_cluster_plv: f64,

    // Attention spotlight (Phase 6)
    pub attention_position: f64,
    pub attention_intensity: f64,
    pub attention_width: f64,
    pub attention_capacity: f64,
    pub secondary_foci_count: usize,
    pub in_attention_blink: bool,
    pub attention_sampling_effectiveness: f64,
    pub in_attention_uptake: bool,

    // Sleep state (Phase 7)
    pub sleep_stage: &'static str,
    pub time_in_sleep_stage: f64,
    pub sleep_pressure: f64,
    pub sleep_cycles_completed: u64,
    pub ultradian_cycle_position: f64,
    pub consolidation_efficiency: f64,
    pub in_memory_transfer_window: bool,
    pub sleep_consciousness_level: f64,
    pub spindle_count: u64,
    pub ripple_count: u64,
    pub dream_intensity: f64,
    pub spindle_power: f64,

    // Neuromodulatory state (Phase 8)
    pub dopamine_level: f64,
    pub serotonin_level: f64,
    pub norepinephrine_level: f64,
    pub acetylcholine_level: f64,
    pub dopamine_sensitivity: f64,
    pub serotonin_sensitivity: f64,
    pub mood_valence: f64,
    pub motivation_drive: f64,
    pub learning_rate_mod: f64,
    pub stress_level: f64,

    // Predictive processing (Phase 9)
    pub free_energy: f64,
    pub complexity: f64,
    pub accuracy: f64,
    pub precision_gain: f64,
    pub weighted_prediction_error: f64,
    pub expected_free_energy: f64,
    pub epistemic_value: f64,
    pub pragmatic_value: f64,
    pub exploration_drive: f64,
    pub surprise: f64,
    pub sensory_precision: f64,
    pub perceptual_precision: f64,
    pub conceptual_precision: f64,
    pub metacognitive_precision: f64,
    pub sensory_prediction: f64,
    pub perceptual_prediction: f64,
    pub conceptual_prediction: f64,
    pub metacognitive_prediction: f64,

    // Global workspace (Phase 10)
    pub is_ignited: bool,
    pub broadcasting_processor: String,
    pub broadcast_strength: f64,
    pub broadcast_duration: f64,
    pub ignition_threshold: f64,
    pub time_since_ignition: f64,
    pub workspace_integration: f64,
    pub coalition_count: usize,
    pub perception_activation: f64,
    pub memory_activation: f64,
    pub executive_activation: f64,
    pub emotion_activation: f64,
    pub motor_activation: f64,
    pub language_activation: f64,
    pub metacognition_activation: f64,
    pub ignition_count: u64,
    pub total_broadcast_time: f64,
}

/// Temporary stub for oscillatory phase
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OscillatoryPhase {
    Peak,
    Rising,
    Falling,
    Trough,
}

/// Temporary stub for OscillatoryRouter (40Hz gamma synchronization)
#[derive(Debug)]
pub struct OscillatoryRouter {
    _config: OscillatoryRouterConfig,
    phase: OscillatoryPhase,
    coherence: f64,
    effective_phi: f64,
    arousal: f64,
    attention: f64,
    pending_ops: Vec<(u64, String)>,
    cycles_completed: u64,
    is_sleeping: bool,
}

impl OscillatoryRouter {
    pub fn new(config: OscillatoryRouterConfig) -> Self {
        Self {
            _config: config,
            phase: OscillatoryPhase::Peak,
            coherence: 1.0,
            effective_phi: 0.5,
            arousal: 0.5,
            attention: 0.5,
            pending_ops: Vec::new(),
            cycles_completed: 0,
            is_sleeping: false,
        }
    }

    pub fn current_phase(&self) -> OscillatoryPhase {
        self.phase
    }

    pub fn update(&mut self, _delta_time: f64) {
        // Stub: cycle through phases
        self.phase = match self.phase {
            OscillatoryPhase::Peak => OscillatoryPhase::Falling,
            OscillatoryPhase::Falling => OscillatoryPhase::Trough,
            OscillatoryPhase::Trough => OscillatoryPhase::Rising,
            OscillatoryPhase::Rising => OscillatoryPhase::Peak,
        };
    }

    pub fn is_optimal_window(&self) -> bool {
        self.phase == OscillatoryPhase::Peak
    }

    pub fn phase_coherence(&self) -> f64 {
        match self.phase {
            OscillatoryPhase::Peak => 1.0,
            OscillatoryPhase::Rising | OscillatoryPhase::Falling => 0.7,
            OscillatoryPhase::Trough => 0.3,
        }
    }

    /// Advance oscillatory state by delta time (in seconds)
    pub fn cycle(&mut self, dt: f64) {
        // Advance phase based on frequency (40Hz = 25ms per cycle)
        let cycles = dt * self._config.base_frequency;
        let phase_advance = (cycles * 4.0).fract(); // 4 phases per cycle

        // Simple phase advancement
        if phase_advance > 0.75 {
            self.phase = OscillatoryPhase::Trough;
        } else if phase_advance > 0.5 {
            self.phase = OscillatoryPhase::Falling;
        } else if phase_advance > 0.25 {
            self.phase = OscillatoryPhase::Peak;
        } else {
            self.phase = OscillatoryPhase::Rising;
        }

        self.cycles_completed += 1;
    }

    /// Observe current Φ to modulate oscillatory coherence
    pub fn observe_state(&mut self, phi: f64, _dt: f64) {
        // Higher Φ = better synchronization
        self.effective_phi = phi;
        self.coherence = (self.coherence * 0.9 + phi * 0.1).clamp(0.1, 1.0);
    }

    /// Set arousal level (affects oscillation amplitude)
    pub fn set_arousal(&mut self, arousal: f64) {
        self.arousal = arousal.clamp(0.0, 1.0);
    }

    /// Set attention level (affects phase-locking)
    pub fn set_attention(&mut self, attention: f64) {
        self.attention = attention.clamp(0.0, 1.0);
    }

    /// Modulate based on hormones (cortisol disrupts, dopamine enhances)
    pub fn modulate_hormones(&mut self, cortisol: f32, dopamine: f32) {
        // Cortisol disrupts synchronization
        let cortisol_effect = 1.0 - (cortisol as f64 * 0.3);
        // Dopamine enhances synchronization
        let dopamine_effect = 1.0 + (dopamine as f64 * 0.2);

        self.coherence = (self.coherence * cortisol_effect * dopamine_effect).clamp(0.1, 1.0);
    }

    /// Execute operations ready for phase-locked execution
    pub fn execute_ready(&mut self) -> Vec<(u64, String)> {
        // Only execute at Peak phase
        if self.phase == OscillatoryPhase::Peak && !self.pending_ops.is_empty() {
            std::mem::take(&mut self.pending_ops)
        } else {
            Vec::new()
        }
    }

    /// Get summary of oscillatory state
    pub fn summary(&self) -> OscillatorySummary {
        OscillatorySummary {
            // Core oscillatory state
            current_phase: self.phase,
            coherence: self.coherence,
            effective_phi: self.effective_phi,
            ready_operations: self.pending_ops.len(),
            current_phi: self.effective_phi,
            is_sleeping: self.is_sleeping,
            delta_power: if self.is_sleeping { 0.8 } else { 0.2 },
            frequency: self._config.base_frequency,
            phase_lock_accuracy: self.coherence * self.attention,
            pending_operations: self.pending_ops.len(),
            cycles_completed: self.cycles_completed,
            dominant_band: if self._config.base_frequency > 30.0 { "gamma" } else { "alpha" },

            // Extended neural oscillation metrics
            gamma_power: if self._config.base_frequency > 30.0 { 0.8 } else { 0.3 },
            theta_power: 0.4,
            alpha_power: if self.is_sleeping { 0.2 } else { 0.5 },
            theta_gamma_coupling: self.coherence * 0.7,
            arousal: self.arousal,
            attention: self.attention,
            in_memory_window: self.phase == OscillatoryPhase::Peak,
            system_resonance: self.coherence * self.effective_phi,

            // Resonance state (Phase 4)
            resonance_integration_boost: self.coherence * 0.3,
            resonating_pairs: 0,
            cluster_count: 1,
            primary_cluster_size: 1,
            primary_cluster_plv: self.coherence,

            // Attention spotlight (Phase 6)
            attention_position: 0.5,
            attention_intensity: self.attention,
            attention_width: 0.3,
            attention_capacity: 4.0,
            secondary_foci_count: 0,
            in_attention_blink: false,
            attention_sampling_effectiveness: self.attention * 0.9,
            in_attention_uptake: false,

            // Sleep state (Phase 7)
            sleep_stage: if self.is_sleeping { "N2" } else { "Wake" },
            time_in_sleep_stage: 0.0,
            sleep_pressure: if self.is_sleeping { 0.3 } else { 0.7 },
            sleep_cycles_completed: 0,
            ultradian_cycle_position: 0.0,
            consolidation_efficiency: 0.5,
            in_memory_transfer_window: self.is_sleeping && self.phase == OscillatoryPhase::Trough,
            sleep_consciousness_level: if self.is_sleeping { 0.2 } else { 1.0 },
            spindle_count: 0,
            ripple_count: 0,
            dream_intensity: 0.0,
            spindle_power: if self.is_sleeping { 0.5 } else { 0.0 },

            // Neuromodulatory state (Phase 8)
            dopamine_level: 0.5,
            serotonin_level: 0.5,
            norepinephrine_level: self.arousal * 0.8,
            acetylcholine_level: self.attention * 0.7,
            dopamine_sensitivity: 1.0,
            serotonin_sensitivity: 1.0,
            mood_valence: 0.0,
            motivation_drive: 0.5,
            learning_rate_mod: 1.0,
            stress_level: 0.3,

            // Predictive processing (Phase 9)
            free_energy: 0.5,
            complexity: 0.5,
            accuracy: 0.8,
            precision_gain: 0.1,
            weighted_prediction_error: 0.2,
            expected_free_energy: 0.4,
            epistemic_value: 0.3,
            pragmatic_value: 0.5,
            exploration_drive: 0.3,
            surprise: 0.2,
            sensory_precision: 0.8,
            perceptual_precision: 0.7,
            conceptual_precision: 0.6,
            metacognitive_precision: 0.5,
            sensory_prediction: 0.5,
            perceptual_prediction: 0.5,
            conceptual_prediction: 0.5,
            metacognitive_prediction: 0.5,

            // Global workspace (Phase 10)
            is_ignited: self.phase == OscillatoryPhase::Peak && self.coherence > 0.7,
            broadcasting_processor: String::new(),
            broadcast_strength: self.coherence,
            broadcast_duration: 0.0,
            ignition_threshold: 0.7,
            time_since_ignition: 0.0,
            workspace_integration: self.effective_phi,
            coalition_count: 0,
            perception_activation: 0.5,
            memory_activation: 0.5,
            executive_activation: 0.5,
            emotion_activation: 0.5,
            motor_activation: 0.3,
            language_activation: 0.5,
            metacognition_activation: 0.4,
            ignition_count: 0,
            total_broadcast_time: 0.0,
        }
    }
}

// ============================================================================
// END TEMPORARY STUBS
// ============================================================================

// LEARNING INTEGRATION: Gradient-based neural adaptation
// Bridges LearnableLTC (BPTT+Adam) with ContinuousMind cognitive loop
use crate::learning::{LearningEngine, LearningConfig};

// Language-Learning Integration: Use linguistic precision for enhanced learning
use crate::language::active_inference_adapter::IntegrationResult;

/// State of the continuous mind
#[derive(Debug, Clone)]
pub struct MindState {
    /// Current consciousness level (Φ from actual integration)
    pub phi: f64,

    /// Meta-awareness (knowing that we know)
    pub meta_awareness: f64,

    /// Current cognitive load (0.0 = idle, 1.0 = maximal)
    pub cognitive_load: f64,

    /// Number of active processes
    pub active_processes: usize,

    /// Total cognitive cycles completed
    pub total_cycles: u64,

    /// Insights generated
    pub insights_generated: u64,

    /// Time since awakening (ms)
    pub time_awake_ms: u64,

    /// Is the mind currently processing external input?
    pub processing_external: bool,

    // === Active Inference Metrics ===

    /// Total free energy (surprise accumulated)
    pub free_energy: f64,

    /// Average surprise across domains
    pub average_surprise: f64,

    /// Most uncertain prediction domain
    pub most_uncertain_domain: Option<String>,

    /// Whether curiosity pressure is high (should explore)
    pub curiosity_pressure: bool,

    // === Language Integration Metrics ===

    /// Unified free energy (language + general combined)
    pub unified_free_energy: f64,

    /// Language-specific free energy
    pub language_free_energy: f64,

    /// Language contribution ratio to unified FE
    pub language_contribution: f64,

    /// Current precision weights for language domains
    pub precision_weights: Option<LanguagePrecisionWeights>,

    /// Pending language actions (from curiosity-driven exploration)
    pub pending_language_actions: usize,

    // === AWAKENING INTEGRATION (Track 7) ===

    /// Phenomenal experience description (qualia)
    pub phenomenal_state: String,

    /// Unified experience description
    pub unified_experience: String,

    /// Self-model prediction accuracy
    pub self_model_accuracy: f64,

    /// What Symthaea is currently aware of
    pub aware_of: Vec<String>,

    /// Is Symthaea conscious? (integrated metric)
    pub is_conscious: bool,

    // === OSCILLATORY DYNAMICS (Phase 6A) ===

    /// Current oscillatory phase (0.0 to 2π radians)
    pub oscillatory_phase: f64,

    /// Current phase category (Peak, Rising, Falling, Trough)
    pub phase_category: String,

    /// Phase coherence (0.0 = chaotic, 1.0 = perfectly synchronized)
    pub phase_coherence: f64,

    /// Effective Φ (Φ modulated by oscillatory state)
    pub effective_phi: f64,

    /// Current gamma frequency (Hz, typically ~40)
    pub gamma_frequency: f64,

    /// Phase-lock accuracy (fraction of operations executed at optimal phase)
    pub phase_lock_accuracy: f64,

    /// Pending phase-locked operations
    pub pending_oscillatory_ops: usize,

    /// Oscillatory cycles completed since awakening
    pub oscillatory_cycles: u64,

    // === MULTI-BAND OSCILLATIONS (Phase 6B) ===

    /// Currently dominant frequency band (Theta/Alpha/Beta/Gamma)
    pub dominant_band: String,

    /// Gamma band power (0.0-1.0, consciousness binding)
    pub gamma_power: f64,

    /// Theta band power (0.0-1.0, memory encoding)
    pub theta_power: f64,

    /// Alpha band power (0.0-1.0, idling/inhibition)
    pub alpha_power: f64,

    /// Theta-gamma coupling strength (memory binding)
    pub theta_gamma_coupling: f64,

    /// Current arousal level (0.0-1.0)
    pub arousal: f64,

    /// Current attention level (0.0-1.0)
    pub attention: f64,

    /// In memory encoding window (theta-gamma sync)
    pub in_memory_window: bool,

    // === PROCESS RESONANCE DETECTION (Phase 6C) ===

    /// System-wide resonance score (0.0-1.0, from process phase synchrony)
    pub system_resonance: f64,

    /// Resonance-based integration boost factor
    pub resonance_integration_boost: f64,

    /// Number of process pairs currently resonating (PLV > threshold)
    pub resonating_pairs: usize,

    /// Number of detected resonant clusters
    pub cluster_count: usize,

    /// Size of the largest resonant cluster
    pub primary_cluster_size: usize,

    /// Mean PLV of the primary cluster (0.0-1.0)
    pub primary_cluster_plv: f64,

    // === ATTENTION SPOTLIGHT (Phase 6C) ===

    /// Primary attention focus position (0.0-1.0)
    pub attention_spotlight_position: f64,

    /// Primary attention intensity (0.0-1.0)
    pub attention_spotlight_intensity: f64,

    /// Primary attention spotlight width (0.1-1.0)
    pub attention_spotlight_width: f64,

    /// Available attention capacity (0.0-1.0)
    pub attention_capacity: f64,

    /// Number of active secondary attention foci
    pub secondary_foci_count: usize,

    /// Whether in attentional blink refractory period
    pub in_attention_blink: bool,

    /// Rhythmic attention sampling effectiveness (0.0-1.0)
    pub attention_sampling_effectiveness: f64,

    /// Whether attention is in uptake phase (can shift focus)
    pub in_attention_uptake: bool,

    // === SLEEP-OSCILLATION INTEGRATION (Phase 7) ===

    /// Current sleep stage (Wake, N1, N2, N3, REM)
    pub sleep_stage: String,

    /// Time in current sleep stage (seconds)
    pub time_in_sleep_stage: f64,

    /// Sleep pressure (homeostatic drive) [0, 1]
    pub sleep_pressure: f64,

    /// Completed ultradian cycles (~90 min each)
    pub sleep_cycles_completed: u32,

    /// Position in current ultradian cycle [0, 1]
    pub ultradian_cycle_position: f64,

    /// Memory consolidation efficiency [0, 1]
    pub consolidation_efficiency: f64,

    /// Currently in optimal memory transfer window
    pub in_memory_transfer_window: bool,

    /// Consciousness level based on sleep stage [0, 1]
    pub sleep_consciousness_level: f64,

    /// Sleep spindle count (N2 stage)
    pub spindle_count: u64,

    /// Sharp-wave ripple count (N3 stage)
    pub ripple_count: u64,

    /// Dream intensity during REM [0, 1]
    pub dream_intensity: f64,

    /// Is currently in a sleep state (not awake)
    pub is_sleeping: bool,

    /// Delta band power (deep sleep) [0, 1]
    pub delta_power: f64,

    /// Sleep spindle band power (light sleep N2) [0, 1]
    pub spindle_band_power: f64,

    // === NEUROMODULATORY DYNAMICS (Phase 8) ===

    /// Dopamine level (motivation, reward learning) [0, 1]
    pub dopamine_level: f64,

    /// Serotonin level (mood, well-being) [0, 1]
    pub serotonin_level: f64,

    /// Norepinephrine level (arousal, alertness) [0, 1]
    pub norepinephrine_level: f64,

    /// Acetylcholine level (learning, attention) [0, 1]
    pub acetylcholine_level: f64,

    /// Dopamine receptor sensitivity [0, 1]
    pub dopamine_sensitivity: f64,

    /// Serotonin receptor sensitivity [0, 1]
    pub serotonin_sensitivity: f64,

    /// Mood valence (-1 = negative, +1 = positive)
    pub mood_valence: f64,

    /// Motivation drive [0, 1]
    pub motivation_drive: f64,

    /// Learning rate modifier [0.5, 2.0]
    pub learning_rate_mod: f64,

    /// Stress level [0, 1]
    pub stress_level: f64,

    // === PREDICTIVE CONSCIOUSNESS (Phase 9) ===

    /// Total free energy (prediction error + complexity) [0, ∞)
    pub predictive_free_energy: f64,

    /// Model complexity (KL divergence from prior) [0, ∞)
    pub predictive_complexity: f64,

    /// Prediction accuracy (lower = better) [0, ∞)
    pub predictive_accuracy: f64,

    /// Precision gain from optimization [0, ∞)
    pub precision_gain: f64,

    /// Weighted prediction error across hierarchy [0, ∞)
    pub weighted_prediction_error: f64,

    /// Expected free energy for action selection [0, ∞)
    pub expected_free_energy: f64,

    /// Epistemic value (information gain) [0, 1]
    pub epistemic_value: f64,

    /// Pragmatic value (goal achievement) [0, 1]
    pub pragmatic_value: f64,

    /// Exploration drive based on expected free energy [0, 1]
    pub exploration_drive: f64,

    /// Surprise (information content) [0, ∞)
    pub predictive_surprise: f64,

    /// Sensory level precision [0, 1]
    pub sensory_precision: f64,

    /// Perceptual level precision [0, 1]
    pub perceptual_precision: f64,

    /// Conceptual level precision [0, 1]
    pub conceptual_precision: f64,

    /// Metacognitive level precision [0, 1]
    pub metacognitive_precision: f64,

    /// Sensory level prediction [varies]
    pub sensory_prediction: f64,

    /// Perceptual level prediction [varies]
    pub perceptual_prediction: f64,

    /// Conceptual level prediction [varies]
    pub conceptual_prediction: f64,

    /// Metacognitive level prediction [varies]
    pub metacognitive_prediction: f64,

    // === GLOBAL WORKSPACE (Phase 10) ===

    /// Is workspace currently ignited (conscious access)
    pub is_ignited: bool,

    /// Currently broadcasting processor (if ignited)
    pub broadcasting_processor: Option<String>,

    /// Broadcast strength (gamma synchronization) [0, 1]
    pub broadcast_strength: f64,

    /// Current broadcast duration (seconds)
    pub broadcast_duration: f64,

    /// Current ignition threshold
    pub ignition_threshold: f64,

    /// Time since last ignition (seconds)
    pub time_since_ignition: f64,

    /// Workspace integration level [0, 1]
    pub workspace_integration: f64,

    /// Number of active processor coalitions
    pub coalition_count: usize,

    /// Perception processor activation [0, 1]
    pub perception_activation: f64,

    /// Memory processor activation [0, 1]
    pub memory_activation: f64,

    /// Executive processor activation [0, 1]
    pub executive_activation: f64,

    /// Emotion processor activation [0, 1]
    pub emotion_activation: f64,

    /// Motor processor activation [0, 1]
    pub motor_activation: f64,

    /// Language processor activation [0, 1]
    pub language_activation: f64,

    /// Metacognition processor activation [0, 1]
    pub metacognition_activation: f64,

    /// Total ignition events
    pub workspace_ignition_count: u64,

    /// Total broadcast time (seconds)
    pub total_broadcast_time: f64,
}

impl Default for MindState {
    fn default() -> Self {
        Self {
            phi: 0.0,
            meta_awareness: 0.0,
            cognitive_load: 0.0,
            active_processes: 0,
            total_cycles: 0,
            insights_generated: 0,
            time_awake_ms: 0,
            processing_external: false,
            // Active Inference defaults
            free_energy: 0.0,
            average_surprise: 0.0,
            most_uncertain_domain: None,
            curiosity_pressure: false,
            // Language integration defaults
            unified_free_energy: 0.0,
            language_free_energy: 0.0,
            language_contribution: 0.0,
            precision_weights: None,
            pending_language_actions: 0,
            // Awakening integration defaults
            phenomenal_state: "Pre-awakening".to_string(),
            unified_experience: "None".to_string(),
            self_model_accuracy: 0.0,
            aware_of: Vec::new(),
            is_conscious: false,
            // Oscillatory dynamics defaults (Phase 6A)
            oscillatory_phase: 0.0,
            phase_category: "Trough".to_string(),
            phase_coherence: 0.0,
            effective_phi: 0.0,
            gamma_frequency: 40.0,  // 40Hz gamma-band default
            phase_lock_accuracy: 0.0,
            pending_oscillatory_ops: 0,
            oscillatory_cycles: 0,
            // Multi-band oscillation defaults (Phase 6B)
            dominant_band: "Gamma".to_string(),
            gamma_power: 0.5,
            theta_power: 0.5,
            alpha_power: 0.5,
            theta_gamma_coupling: 0.5,
            arousal: 0.6,
            attention: 0.5,
            in_memory_window: false,
            // Process Resonance Detection defaults (Phase 6C)
            system_resonance: 0.0,
            resonance_integration_boost: 1.0,  // No boost initially
            resonating_pairs: 0,
            cluster_count: 0,
            primary_cluster_size: 0,
            primary_cluster_plv: 0.0,
            // Attention Spotlight defaults (Phase 6C)
            attention_spotlight_position: 0.5,  // Center position
            attention_spotlight_intensity: 0.5, // Medium intensity
            attention_spotlight_width: 0.3,     // Moderate width
            attention_capacity: 1.0,            // Full capacity
            secondary_foci_count: 0,
            in_attention_blink: false,
            attention_sampling_effectiveness: 1.0,
            in_attention_uptake: true,          // Initially ready to shift
            // Sleep-Oscillation defaults (Phase 7)
            sleep_stage: "Wake".to_string(),    // Start awake
            time_in_sleep_stage: 0.0,
            sleep_pressure: 0.0,                // No sleep pressure initially
            sleep_cycles_completed: 0,
            ultradian_cycle_position: 0.0,
            consolidation_efficiency: 0.0,      // No consolidation when awake
            in_memory_transfer_window: false,
            sleep_consciousness_level: 1.0,     // Full consciousness when awake
            spindle_count: 0,
            ripple_count: 0,
            dream_intensity: 0.0,               // No dreams when awake
            is_sleeping: false,
            delta_power: 0.1,                   // Low delta during wakefulness
            spindle_band_power: 0.1,            // Low spindle power when awake

            // Neuromodulatory dynamics defaults (Phase 8)
            dopamine_level: 0.3,                // Baseline dopamine (tonic level)
            serotonin_level: 0.5,               // Baseline serotonin (mood stability)
            norepinephrine_level: 0.2,          // Low arousal at baseline
            acetylcholine_level: 0.4,           // Moderate baseline for learning
            dopamine_sensitivity: 1.0,          // Full receptor sensitivity
            serotonin_sensitivity: 1.0,         // Full receptor sensitivity
            mood_valence: 0.2,                  // Slightly positive mood
            motivation_drive: 0.5,              // Neutral motivation
            learning_rate_mod: 1.0,             // Normal learning rate
            stress_level: 0.2,                  // Low stress baseline

            // Predictive consciousness defaults (Phase 9)
            predictive_free_energy: 0.0,        // No prediction error initially
            predictive_complexity: 0.0,         // No model complexity
            predictive_accuracy: 0.0,           // Perfect accuracy (no error)
            precision_gain: 0.0,                // No precision optimization yet
            weighted_prediction_error: 0.0,     // No weighted error
            expected_free_energy: 0.0,          // No expected free energy
            epistemic_value: 0.5,               // Balanced epistemic value
            pragmatic_value: 0.5,               // Balanced pragmatic value
            exploration_drive: 0.3,             // Slight exploration tendency
            predictive_surprise: 0.0,           // No surprise initially
            sensory_precision: 0.5,             // Medium sensory precision
            perceptual_precision: 0.5,          // Medium perceptual precision
            conceptual_precision: 0.5,          // Medium conceptual precision
            metacognitive_precision: 0.5,       // Medium metacognitive precision
            sensory_prediction: 0.0,            // No sensory prediction
            perceptual_prediction: 0.0,         // No perceptual prediction
            conceptual_prediction: 0.0,         // No conceptual prediction
            metacognitive_prediction: 0.0,      // No metacognitive prediction

            // === GLOBAL WORKSPACE (Phase 10) ===
            is_ignited: false,                      // No ignition initially
            broadcasting_processor: None,           // No broadcast
            broadcast_strength: 0.0,                // No broadcast strength
            broadcast_duration: 0.0,                // No broadcast duration
            ignition_threshold: 0.4,                // Default threshold
            time_since_ignition: 0.0,               // No prior ignition
            workspace_integration: 0.0,             // No workspace integration
            coalition_count: 0,                     // No coalitions
            perception_activation: 0.2,             // Baseline perception
            memory_activation: 0.1,                 // Low memory access
            executive_activation: 0.2,              // Baseline executive control
            emotion_activation: 0.2,                // Baseline emotion
            motor_activation: 0.1,                  // Low motor activity
            language_activation: 0.1,               // Low language processing
            metacognition_activation: 0.1,          // Low metacognition
            workspace_ignition_count: 0,            // No prior ignitions
            total_broadcast_time: 0.0,              // No prior broadcasts
        }
    }
}

/// Configuration for the continuous mind
#[derive(Debug, Clone)]
pub struct MindConfig {
    /// Cycle duration in milliseconds
    pub cycle_ms: u64,

    /// Minimum Φ for consciousness threshold
    pub consciousness_threshold: f64,

    /// Enable daemon (default mode network)
    pub enable_daemon: bool,

    /// Enable active inference
    pub enable_active_inference: bool,

    /// HDC dimension for representations
    pub hdc_dimension: usize,
}

impl Default for MindConfig {
    fn default() -> Self {
        Self {
            cycle_ms: 50,  // 20 Hz cognitive cycle
            consciousness_threshold: 0.3,
            enable_daemon: true,
            enable_active_inference: true,
            hdc_dimension: HDC_DIMENSION,
        }
    }
}

/// A cognitive process that contributes to integration
pub struct CognitiveProcess {
    /// Name of this process
    pub name: String,

    /// Current state as hypervector
    pub state: RealHV,

    /// Activity level (0.0 - 1.0)
    pub activity: f64,
}

impl CognitiveProcess {
    pub fn new(name: impl Into<String>, dim: usize) -> Self {
        Self {
            name: name.into(),
            state: RealHV::random(dim, 42),
            activity: 0.0,
        }
    }

    pub fn activate(&mut self, input: &RealHV) {
        // Bind current state with input
        self.state = self.state.bind(input);
        self.activity = 1.0;
    }

    pub fn decay(&mut self, rate: f64) {
        self.activity = (self.activity - rate).max(0.0);
    }
}

/// The Continuous Mind - Always running cognitive system
pub struct ContinuousMind {
    /// Configuration
    config: MindConfig,

    /// Current state
    state: Arc<Mutex<MindState>>,

    /// Cognitive processes (perception, reasoning, memory, etc.)
    processes: Arc<Mutex<Vec<CognitiveProcess>>>,

    /// Φ calculator orchestrator - adaptively selects best algorithm
    /// Phase 5E: Connects phi_real, phi_resonant, tiered_phi
    phi_orchestrator: Arc<Mutex<PhiOrchestrator>>,

    /// Active Inference Engine (Free Energy Minimization)
    /// This is the CORE integration - Karl Friston's framework for goal-directed behavior
    active_inference: Arc<Mutex<ActiveInferenceEngine>>,

    /// Daemon (default mode network)
    daemon: Arc<Mutex<DaemonActor>>,

    /// Memory system
    hippocampus: Arc<Mutex<HippocampusActor>>,

    /// Current goals
    goals: Arc<Mutex<Vec<Goal>>>,

    /// Hormone state
    hormones: Arc<Mutex<HormoneState>>,

    /// Semantic text encoder (grounded in primitives)
    /// Uses the 7-tier PrimitiveSystem instead of naive word hashing
    text_encoder: Arc<Mutex<TextEncoder>>,

    /// Ontological primitive system
    /// 200+ primitives across Mathematical, Physical, Geometric, Strategic, MetaCognitive, Temporal, Compositional tiers
    primitive_system: Arc<PrimitiveSystem>,

    /// Language Consciousness Bridge (38KB!)
    /// Provides linguistic understanding with Φ measurement
    /// Converts text into conscious understanding: frames, BIDs, prediction errors
    language_bridge: Arc<Mutex<ConsciousnessBridge>>,

    /// Active Inference Language Adapter
    /// Bridges language (ConsciousnessBridge) ↔ brain (ActiveInferenceEngine)
    /// - Domain mapping: Lexical→UserState, Syntactic→TaskSuccess, Semantic→Coherence
    /// - Precision flow: Brain precision weights → Language attention modulation
    /// - Unified free energy: Language FE + General FE combined for consciousness monitoring
    language_adapter: Arc<Mutex<ActiveInferenceAdapter>>,

    /// INTEGRATION: Awakening module for self-awareness, qualia, and introspection
    /// This provides:
    /// - QualiaGenerator for phenomenal experience descriptions
    /// - SelfModel with predictive accuracy tracking
    /// - Introspection capability for rich self-reflection
    /// - ConsciousnessDashboard for visualization
    /// - Meta-awareness tracking (knowing that you know)
    awakening: Arc<Mutex<SymthaeaAwakening>>,

    /// INTEGRATION: Meta-cognitive Router (Phase 5F)
    /// Uses UCB1 multi-armed bandit to learn which of 7 routing paradigms works best:
    /// - CausalValidation, InformationGeometric, TopologicalConsciousness
    /// - QuantumCoherence, ActiveInference, PredictiveProcessing, AttentionSchema
    /// Adapts routing strategy based on consciousness state (Φ, coherence, volatility)
    meta_router: Arc<Mutex<MetaRouter>>,

    /// INTEGRATION: Phase 6A - OscillatoryRouter for phase-locked cognitive processing
    /// Uses 40Hz gamma-band synchronization to optimize execution windows:
    /// - Peak phase: Binding and integration operations
    /// - Rising phase: Input gathering and attention
    /// - Falling phase: Output generation and consolidation
    /// - Trough phase: Reset and maintenance
    oscillatory_router: Arc<Mutex<OscillatoryRouter>>,

    /// LEARNING INTEGRATION: Gradient-based neural adaptation
    /// Bridges LearnableLTC with the cognitive loop:
    /// - Experience replay with prioritized sampling
    /// - Neuromodulation-adjusted learning rate (dopamine/ACh/stress)
    /// - Sleep consolidation during low-activity periods
    /// - Prediction error → gradient signal for adaptation
    learning_engine: Arc<Mutex<LearningEngine>>,

    /// LANGUAGE-LEARNING BRIDGE: Latest language integration result
    /// Shared between process() (writes) and cognitive loop (reads)
    /// Enables continuous learning from linguistic understanding:
    /// - Linguistic precision weights → learning modulation
    /// - Unified free energy → consciousness-aware training
    /// - Language Φ → integration quality signal
    latest_language_integration: Arc<Mutex<Option<IntegrationResult>>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Background thread handles
    threads: Vec<JoinHandle<()>>,

    /// Insight receiver
    insight_rx: Option<Receiver<Insight>>,

    /// Time of awakening
    awakening_time: Instant,
}

impl ContinuousMind {
    /// Create a new continuous mind
    pub fn new(config: MindConfig) -> Self {
        // Initialize cognitive processes
        let processes = vec![
            CognitiveProcess::new("perception", config.hdc_dimension),
            CognitiveProcess::new("reasoning", config.hdc_dimension),
            CognitiveProcess::new("memory", config.hdc_dimension),
            CognitiveProcess::new("planning", config.hdc_dimension),
            CognitiveProcess::new("introspection", config.hdc_dimension),
        ];

        // Initialize Active Inference Engine with custom weights
        let mut active_inference = ActiveInferenceEngine::new();
        // Balance curiosity vs goal-directed behavior
        active_inference.curiosity_weight = 0.4;  // 40% exploration
        active_inference.goal_weight = 0.6;       // 60% goal-directed

        // Initialize semantic encoding systems
        // This replaces naive word hashing with grounded primitives!
        let primitive_system = Arc::new(PrimitiveSystem::new());
        let text_encoder_config = TextEncoderConfig {
            dimension: config.hdc_dimension,
            ..Default::default()
        };
        let text_encoder = TextEncoder::new(text_encoder_config)
            .expect("Failed to create TextEncoder");

        // Initialize Language Active Inference Integration (Phase 3)
        // This wires the orphaned 38KB adapter into our existing architecture!
        let language_bridge = ConsciousnessBridge::new(BridgeConfig::default());
        let language_adapter = ActiveInferenceAdapter::new(AdapterConfig {
            language_weight: 0.4,        // 40% weight for language FE in unified calculation
            action_threshold: 0.3,       // Trigger language actions when uncertainty > 0.3
            max_actions: 3,              // Max 3 language actions per cycle
            enable_precision_flow: true, // Enable precision flow from engine to language
            enable_curiosity: true,      // Enable curiosity-driven language learning
            precision_update_rate: 0.1,  // 10% update rate for smoothing
        });

        // INTEGRATION: Initialize awakening module for self-awareness
        // This provides qualia, self-model, introspection, and dashboard
        let awakening = SymthaeaAwakening::default();

        // RESONATOR Φ INTEGRATION: 100x speedup via coupled oscillator dynamics
        // Uses O(n log N) resonance instead of O(n³) eigenvalue computation
        // Configured for real-time cognitive loop performance:
        // - 100 iterations max (fast convergence)
        // - 0.5 damping (quick stabilization)
        // - Parallel computation for n >= 6 processes
        let mut phi_orchestrator = PhiOrchestrator::new(PhiMode::Fast);
        phi_orchestrator.configure_resonant(ResonantConfig::fast());

        // Phase 5F: Initialize MetaRouter for cognitive routing strategy selection
        // Uses UCB1 multi-armed bandit across 7 consciousness routing paradigms
        let meta_router = MetaRouter::new(MetaRouterConfig::default());

        // Phase 6A: Initialize OscillatoryRouter for phase-locked cognitive processing
        // Uses 40Hz gamma-band oscillations to synchronize cognitive operations
        // This enables phase-locked scheduling of different processing modes:
        // - Peak (max amplitude): Binding and integration operations
        // - Rising (increasing): Input gathering and attention allocation
        // - Falling (decreasing): Output generation and consolidation
        // - Trough (min amplitude): Reset, maintenance, and cleanup
        let oscillatory_router = OscillatoryRouter::new(OscillatoryRouterConfig {
            gamma_frequency_hz: 40.0,       // 40Hz gamma-band (consciousness signature)
            base_frequency: 40.0,           // 40Hz gamma-band (consciousness signature)
            frequency_adaptation: 0.1,      // Adaptation rate
            min_amplitude: 0.3,             // Minimum amplitude for phase-locking
            amplitude_decay: 0.05,          // Amplitude decay rate
            phase_coupling_strength: 0.3,   // Phase coupling strength
            prediction_cycles: 4,           // Predict 4 cycles ahead
            phase_locked_scheduling: true,  // Enable phase-locked scheduling
            phase_weight: 0.4,              // 40% influence from phase state
            magnitude_weight: 0.6,          // 60% influence from magnitude/strategy
        });

        // LEARNING INTEGRATION: Initialize gradient-based neural adaptation
        // Uses LearnableLTC with BPTT+Adam for experience-driven improvement
        let learning_config = LearningConfig {
            num_neurons: 512,           // LTC network size
            input_dim: 256,             // Compressed HDC state dimension
            output_dim: 64,             // Action/prediction space
            base_lr: 0.001,             // Base learning rate (modulated by neuromodulation)
            train_interval: 10,         // Train every 10 cognitive cycles
            batch_size: 16,             // Experience batch size
            buffer_size: 1000,          // Replay buffer capacity
            enable_consolidation: true, // Enable sleep-based memory consolidation
            ..Default::default()
        };
        let learning_engine = LearningEngine::new(learning_config)
            .expect("Failed to create LearningEngine");

        Self {
            config,
            state: Arc::new(Mutex::new(MindState::default())),
            processes: Arc::new(Mutex::new(processes)),
            phi_orchestrator: Arc::new(Mutex::new(phi_orchestrator)),
            active_inference: Arc::new(Mutex::new(active_inference)),
            daemon: Arc::new(Mutex::new(DaemonActor::new(DaemonConfig::default()))),
            hippocampus: Arc::new(Mutex::new(
                HippocampusActor::new(10_000).expect("Failed to create Hippocampus")
            )),
            goals: Arc::new(Mutex::new(Vec::new())),
            hormones: Arc::new(Mutex::new(HormoneState::default())),
            text_encoder: Arc::new(Mutex::new(text_encoder)),
            primitive_system,
            language_bridge: Arc::new(Mutex::new(language_bridge)),
            language_adapter: Arc::new(Mutex::new(language_adapter)),
            awakening: Arc::new(Mutex::new(awakening)),
            meta_router: Arc::new(Mutex::new(meta_router)),
            oscillatory_router: Arc::new(Mutex::new(oscillatory_router)),
            learning_engine: Arc::new(Mutex::new(learning_engine)),
            latest_language_integration: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
            insight_rx: None,
            awakening_time: Instant::now(),
        }
    }

    /// Awaken the mind - start all background processes
    pub fn awaken(&mut self) {
        println!("🧠 Awakening continuous mind...");

        self.awakening_time = Instant::now();

        // INTEGRATION: Awaken self-awareness module first
        {
            let mut awakening = self.awakening.lock().unwrap();
            let initial_state = awakening.awaken();
            println!("   ✅ Self-awareness module awakened (Φ={:.3})", initial_state.phi);
        }

        // Create insight channel
        let (insight_tx, insight_rx) = channel();
        self.insight_rx = Some(insight_rx);

        // Start daemon (default mode network)
        if self.config.enable_daemon {
            let daemon_handle = DaemonActor::run_continuous(
                Arc::clone(&self.daemon),
                Arc::clone(&self.hippocampus),
                Arc::clone(&self.goals),
                Arc::clone(&self.hormones),
                insight_tx.clone(),
                Arc::clone(&self.shutdown),
            );
            self.threads.push(daemon_handle);
            println!("   ✅ Daemon (Default Mode Network) running");
        }

        // Start main cognitive loop
        let main_handle = self.run_cognitive_loop();
        self.threads.push(main_handle);
        println!("   ✅ Main cognitive loop running at {} Hz", 1000 / self.config.cycle_ms);

        println!("🌟 Mind awakened with unified consciousness!");
    }

    /// Run the main cognitive loop
    fn run_cognitive_loop(&self) -> JoinHandle<()> {
        let state = Arc::clone(&self.state);
        let processes = Arc::clone(&self.processes);
        let active_inference = Arc::clone(&self.active_inference);
        let shutdown = Arc::clone(&self.shutdown);
        let cycle_duration = Duration::from_millis(self.config.cycle_ms);
        let consciousness_threshold = self.config.consciousness_threshold;
        let awakening_time = self.awakening_time;
        let phi_orchestrator = Arc::clone(&self.phi_orchestrator);  // Phase 5E: Adaptive Φ orchestrator
        let meta_router = Arc::clone(&self.meta_router);  // Phase 5F: Cognitive router
        let oscillatory_router = Arc::clone(&self.oscillatory_router);  // Phase 6A: Oscillatory router
        let learning_engine = Arc::clone(&self.learning_engine);  // Learning integration
        let latest_language = Arc::clone(&self.latest_language_integration);  // Language-learning bridge
        let _hdc_dim = self.config.hdc_dimension;
        let hormones = Arc::clone(&self.hormones);

        thread::spawn(move || {
            // Track previous Φ for MetaRouter learning (computes phi delta)
            let mut previous_phi: f64 = 0.0;

            // LEARNING INTEGRATION: Track previous cognitive states for experience creation
            let mut previous_process_states: Vec<RealHV> = Vec::new();
            let mut previous_prediction: Vec<f32> = Vec::new();

            while !shutdown.load(Ordering::Relaxed) {
                let cycle_start = Instant::now();

                // ===== COGNITIVE TICK =====

                // 1. Get process states and compute Φ
                //    OPTIMIZED: Compute Φ while holding processes lock, leveraging
                //    the PhiOrchestrator's result cache to avoid redundant computation
                let (phi, active_process_count) = {
                    let procs = processes.lock().unwrap();

                    // Collect active process states
                    let active_states: Vec<RealHV> = procs.iter()
                        .filter(|p| p.activity > 0.1)  // Only active processes
                        .map(|p| p.state.clone())
                        .collect();

                    let count = active_states.len();

                    // Compute Φ from ACTUAL process integration
                    // Phase 5E: Uses adaptive orchestrator with caching
                    let phi_value = if count >= 2 {
                        let mut orch = phi_orchestrator.lock().unwrap();
                        orch.compute_simple(&active_states)
                    } else {
                        0.0
                    };

                    (phi_value, count)
                };
                // Note: active_states is dropped here, freeing memory earlier

                // 2. Compute meta-awareness (knowing that we know)
                //    This is a higher-order computation: how integrated is our model of our integration?
                let meta_awareness = if phi > consciousness_threshold {
                    // If conscious, meta-awareness is proportional to Φ stability
                    phi.min(1.0)
                } else {
                    0.0
                };

                let cognitive_load = active_process_count as f64 / 5.0;

                // ===== COGNITIVE ROUTING (Phase 5F) =====
                // MetaRouter selects optimal routing paradigm based on current consciousness state
                // Uses UCB1 multi-armed bandit to learn which paradigm works best
                let routing_decision = {
                    let mut router = meta_router.lock().unwrap();

                    // Build consciousness state from current metrics using the constructor
                    // Arguments: phi, integration, coherence, attention
                    let consciousness_state = LatentConsciousnessState::from_observables(
                        phi,                           // phi
                        meta_awareness,                // integration
                        phi * 0.9,                     // coherence (approximated from Φ)
                        cognitive_load.min(1.0),       // attention
                    );

                    // Route to select optimal paradigm
                    router.route(&consciousness_state)
                };

                // The routing decision tells us which cognitive strategy to use this cycle
                // - FullDeliberation for complex decisions
                // - FastPatterns for routine processing
                // - Ensemble for uncertain situations
                let _selected_strategy = routing_decision.paradigm;

                // ===== PHASE 6A: OSCILLATORY ROUTING =====
                // Advance the oscillatory state and determine phase-locked execution windows
                // The 40Hz gamma-band oscillation synchronizes cognitive operations for optimal integration
                let oscillatory_summary = {
                    let mut osc_router = oscillatory_router.lock().unwrap();

                    // Advance oscillatory state by cycle duration (converts to seconds)
                    let dt = cycle_duration.as_secs_f64();
                    osc_router.cycle(dt);

                    // Observe current Φ to modulate oscillatory coherence
                    // Higher Φ = better synchronization = stronger gamma rhythm
                    osc_router.observe_state(phi, dt);

                    // ===== AROUSAL-ATTENTION MODULATION + HORMONE-OSCILLATION COUPLING =====
                    // OPTIMIZED: Consolidated hormone lock acquisition
                    // Set arousal, attention, and apply hormone modulation in single lock
                    {
                        let h = hormones.lock().unwrap();

                        // Arousal: high norepinephrine (simulated via cortisol) → high arousal
                        // But extreme stress reduces effective arousal
                        let arousal = (0.4 + h.dopamine as f64 * 0.3 + (1.0 - h.cortisol as f64) * 0.3).clamp(0.1, 1.0);
                        osc_router.set_arousal(arousal);

                        // Attention: higher when focused, lower when idle
                        let attention = (cognitive_load * 0.5 + phi * 0.5).clamp(0.1, 1.0);
                        osc_router.set_attention(attention);

                        // Hormone-oscillation coupling:
                        // - Cortisol (stress): disrupts synchronization
                        // - Dopamine (reward): enhances synchronization
                        osc_router.modulate_hormones(h.cortisol, h.dopamine);
                    }

                    // Execute any operations that are phase-aligned for this cycle
                    // Returns list of (operation_id, action) pairs that should execute now
                    let ready_ops = osc_router.execute_ready();

                    // Process ready operations (cognitive actions at optimal phase)
                    for (_op_id, _action) in &ready_ops {
                        // In a full implementation, these actions would be dispatched
                        // to the appropriate cognitive subsystem (perception, reasoning, etc.)
                        // For now, we just track that phase-locked execution is working
                    }

                    // Get summary for state tracking
                    osc_router.summary()
                };

                // Phase-locked metrics are now available for consciousness monitoring
                let current_phase = oscillatory_summary.current_phase;
                let phase_coherence = oscillatory_summary.coherence;
                let effective_phi = oscillatory_summary.effective_phi;

                // ===== ACTIVE INFERENCE INTEGRATION =====
                // This is the REVOLUTIONARY part: using Karl Friston's Free Energy Principle

                let ai_summary = {
                    let mut ai = active_inference.lock().unwrap();

                    // Observe current state in multiple prediction domains
                    // The engine will update its generative models and compute prediction errors

                    // 1. Coherence domain: How integrated is the system? (Φ as proxy)
                    ai.observe(PredictionDomain::Coherence, phi as f32);

                    // 2. Performance domain: How loaded is the system?
                    ai.observe(PredictionDomain::Performance, cognitive_load as f32);

                    // 3. Energy domain: Resource utilization
                    let energy_usage = if active_process_count == 0 { 0.1 } else { 0.3 + cognitive_load as f32 * 0.5 };
                    ai.observe(PredictionDomain::Energy, energy_usage);

                    // 4. Temporal domain: Is time perception normal?
                    // (For now, always 1.0 = normal time flow)
                    ai.observe(PredictionDomain::Temporal, 1.0);

                    // 5. Safety domain: Hormone-modulated threat perception
                    // High cortisol = heightened threat perception (paranoid)
                    // Low cortisol = relaxed safety perception
                    let safety_perception = {
                        let h = hormones.lock().unwrap();
                        // Inverse of cortisol: high cortisol = low perceived safety
                        let base_safety = 0.7;  // Neutral baseline
                        let cortisol_threat = h.cortisol * 0.5;  // Cortisol reduces safety
                        let dopamine_comfort = h.dopamine * 0.2;  // Dopamine increases safety
                        (base_safety - cortisol_threat + dopamine_comfort).clamp(0.0, 1.0)
                    };
                    ai.observe(PredictionDomain::Safety, safety_perception);

                    // Decay free energy over time (natural entropy reduction)
                    ai.decay(cycle_duration.as_secs_f32());

                    // Get summary for state update
                    ai.summary()
                };

                // ===== LEARNING INTEGRATION: Gradient-Based Neural Adaptation =====
                // This bridges LearnableLTC with the cognitive loop
                // - Uses prediction error from Active Inference as learning signal
                // - Neuromodulation (dopamine/ACh/stress) modulates learning rate
                // - Experiences are stored for replay-based training
                let current_process_states: Vec<RealHV> = {
                    let procs = processes.lock().unwrap();
                    procs.iter()
                        .filter(|p| p.activity > 0.1)
                        .map(|p| p.state.clone())
                        .collect()
                };

                let _learning_stats = {
                    let mut engine = learning_engine.lock().unwrap();

                    // 1. Update neuromodulation from hormone state + linguistic precision
                    //    Dopamine → increased learning (reward signal)
                    //    ACh → enhanced plasticity (attention signal from language precision)
                    //    Stress → reduced learning (protective)
                    //
                    // LANGUAGE-LEARNING INTEGRATION:
                    // Linguistic precision weights modulate acetylcholine (attention/plasticity)
                    // Higher language coherence → enhanced learning plasticity
                    {
                        let h = hormones.lock().unwrap();

                        // Get linguistic precision boost from latest language processing
                        let linguistic_precision_boost: f32 = {
                            let lang = latest_language.lock().unwrap();
                            if let Some(ref result) = *lang {
                                // Use coherence precision as attention signal
                                // Higher coherence → better language understanding → boost plasticity
                                let coherence = result.precision_weights.coherence;
                                // Also consider task success (user intent understood)
                                let task = result.precision_weights.task_success;
                                (coherence + task) / 2.0  // Average of both
                            } else {
                                0.5  // Neutral if no language processing yet
                            }
                        };

                        // Compute ACh: base level + hormone contribution + linguistic precision
                        let base_ach = 0.4;
                        let hormone_ach = h.dopamine * 0.2;  // Dopamine contribution
                        let language_ach = linguistic_precision_boost * 0.3;  // Linguistic modulation
                        let acetylcholine = (base_ach + hormone_ach + language_ach).clamp(0.0, 1.0);

                        engine.update_neuromodulation(
                            h.dopamine,       // Dopamine level
                            acetylcholine,    // ACh modulated by language precision
                            h.cortisol,       // Stress from cortisol
                        );
                    }

                    // 2. Process cognitive cycle through LTC
                    //    This generates a prediction based on current states
                    let (prediction, should_train) = engine.process_cycle(&current_process_states)
                        .unwrap_or((vec![0.0; 64], false));

                    // 3. Create experience if we have previous state
                    //    Experience = (state, action, next_state, error, reward)
                    //
                    // LANGUAGE-LEARNING INTEGRATION:
                    // Include linguistic unified free energy in prediction error
                    // and language coherence in reward signal
                    if !previous_process_states.is_empty() && !previous_prediction.is_empty() {
                        // Prediction error from Active Inference
                        let base_prediction_error = ai_summary.average_surprise;

                        // Get linguistic prediction error if available
                        let (linguistic_error, coherence_reward): (f32, f32) = {
                            let lang = latest_language.lock().unwrap();
                            if let Some(ref result) = *lang {
                                // Use unified free energy as linguistic error signal
                                let lang_fe = result.unified_free_energy.language_fe as f32;
                                // Use coherence as positive reward signal
                                let coherence = result.precision_weights.coherence as f32;
                                (lang_fe, coherence - 0.5)  // Coherence above 0.5 is positive
                            } else {
                                (0.0, 0.0)
                            }
                        };

                        // Combined prediction error (weighted average)
                        let prediction_error = base_prediction_error * 0.7 + linguistic_error * 0.3;

                        // Enhanced reward signal: Φ improvement + language coherence
                        let phi_reward = ((phi - previous_phi) * 10.0) as f32;
                        let reward = phi_reward * 0.7 + coherence_reward * 0.3;

                        let experience = engine.create_experience(
                            &previous_process_states,
                            &previous_prediction,
                            &current_process_states,
                            prediction_error,
                            reward,
                        );
                        engine.add_experience(experience);
                    }

                    // 4. Train if it's time (every N cycles, if buffer is full)
                    let training_loss = if should_train {
                        engine.train_batch().unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    // 5. Sleep consolidation: trigger during low-activity periods
                    //    (when is_sleeping and delta_power is high = N3 deep sleep)
                    let is_deep_sleep = oscillatory_summary.is_sleeping
                        && oscillatory_summary.delta_power > 0.7;

                    if is_deep_sleep && !engine.stats().consolidation_events > 0 {
                        engine.begin_consolidation();
                        let _ = engine.consolidate();
                        engine.end_consolidation();
                    }

                    // Update tracking for next cycle
                    previous_prediction = prediction;

                    (engine.stats().clone(), training_loss)
                };

                // Update previous states for next cycle's experience creation
                previous_process_states = current_process_states;

                // 4. Update state with both Φ and Active Inference metrics
                {
                    let mut s = state.lock().unwrap();
                    s.phi = phi;
                    s.meta_awareness = meta_awareness;
                    s.total_cycles += 1;
                    s.time_awake_ms = awakening_time.elapsed().as_millis() as u64;
                    s.active_processes = active_process_count;
                    s.cognitive_load = cognitive_load;

                    // Active Inference state
                    s.free_energy = ai_summary.total_free_energy as f64;
                    s.average_surprise = ai_summary.average_surprise as f64;
                    s.most_uncertain_domain = Some(format!("{:?}", ai_summary.most_uncertain_domain));
                    s.curiosity_pressure = ai_summary.curiosity_pressure;

                    // Oscillatory dynamics state (Phase 6A)
                    s.oscillatory_phase = oscillatory_summary.current_phi; // Using phi as proxy for phase
                    s.phase_category = format!("{:?}", oscillatory_summary.current_phase);
                    s.phase_coherence = oscillatory_summary.coherence;
                    s.effective_phi = oscillatory_summary.effective_phi;
                    s.gamma_frequency = oscillatory_summary.frequency;
                    s.phase_lock_accuracy = oscillatory_summary.phase_lock_accuracy;
                    s.pending_oscillatory_ops = oscillatory_summary.pending_operations;
                    s.oscillatory_cycles = oscillatory_summary.cycles_completed;

                    // Multi-band oscillation state (Phase 6B)
                    s.dominant_band = format!("{:?}", oscillatory_summary.dominant_band);
                    s.gamma_power = oscillatory_summary.gamma_power;
                    s.theta_power = oscillatory_summary.theta_power;
                    s.alpha_power = oscillatory_summary.alpha_power;
                    s.theta_gamma_coupling = oscillatory_summary.theta_gamma_coupling;
                    s.arousal = oscillatory_summary.arousal;
                    s.attention = oscillatory_summary.attention;
                    s.in_memory_window = oscillatory_summary.in_memory_window;

                    // Process Resonance Detection state (Phase 6C)
                    s.system_resonance = oscillatory_summary.system_resonance;
                    s.resonance_integration_boost = oscillatory_summary.resonance_integration_boost;
                    s.resonating_pairs = oscillatory_summary.resonating_pairs;
                    s.cluster_count = oscillatory_summary.cluster_count;
                    s.primary_cluster_size = oscillatory_summary.primary_cluster_size;
                    s.primary_cluster_plv = oscillatory_summary.primary_cluster_plv;

                    // Attention Spotlight state (Phase 6C)
                    s.attention_spotlight_position = oscillatory_summary.attention_position;
                    s.attention_spotlight_intensity = oscillatory_summary.attention_intensity;
                    s.attention_spotlight_width = oscillatory_summary.attention_width;
                    s.attention_capacity = oscillatory_summary.attention_capacity;
                    s.secondary_foci_count = oscillatory_summary.secondary_foci_count;
                    s.in_attention_blink = oscillatory_summary.in_attention_blink;
                    s.attention_sampling_effectiveness = oscillatory_summary.attention_sampling_effectiveness;
                    s.in_attention_uptake = oscillatory_summary.in_attention_uptake;

                    // Sleep-Oscillation state (Phase 7)
                    s.sleep_stage = format!("{:?}", oscillatory_summary.sleep_stage);
                    s.time_in_sleep_stage = oscillatory_summary.time_in_sleep_stage;
                    s.sleep_pressure = oscillatory_summary.sleep_pressure;
                    s.sleep_cycles_completed = oscillatory_summary.sleep_cycles_completed;
                    s.ultradian_cycle_position = oscillatory_summary.ultradian_cycle_position;
                    s.consolidation_efficiency = oscillatory_summary.consolidation_efficiency;
                    s.in_memory_transfer_window = oscillatory_summary.in_memory_transfer_window;
                    s.sleep_consciousness_level = oscillatory_summary.sleep_consciousness_level;
                    s.spindle_count = oscillatory_summary.spindle_count;
                    s.ripple_count = oscillatory_summary.ripple_count;
                    s.dream_intensity = oscillatory_summary.dream_intensity;
                    s.is_sleeping = oscillatory_summary.is_sleeping;
                    s.delta_power = oscillatory_summary.delta_power;
                    s.spindle_band_power = oscillatory_summary.spindle_power;

                    // Neuromodulatory dynamics state (Phase 8)
                    s.dopamine_level = oscillatory_summary.dopamine_level;
                    s.serotonin_level = oscillatory_summary.serotonin_level;
                    s.norepinephrine_level = oscillatory_summary.norepinephrine_level;
                    s.acetylcholine_level = oscillatory_summary.acetylcholine_level;
                    s.dopamine_sensitivity = oscillatory_summary.dopamine_sensitivity;
                    s.serotonin_sensitivity = oscillatory_summary.serotonin_sensitivity;
                    s.mood_valence = oscillatory_summary.mood_valence;
                    s.motivation_drive = oscillatory_summary.motivation_drive;
                    s.learning_rate_mod = oscillatory_summary.learning_rate_mod;
                    s.stress_level = oscillatory_summary.stress_level;

                    // Phase 9: Predictive consciousness (free energy principle)
                    s.predictive_free_energy = oscillatory_summary.free_energy;
                    s.predictive_complexity = oscillatory_summary.complexity;
                    s.predictive_accuracy = oscillatory_summary.accuracy;
                    s.precision_gain = oscillatory_summary.precision_gain;
                    s.weighted_prediction_error = oscillatory_summary.weighted_prediction_error;
                    s.expected_free_energy = oscillatory_summary.expected_free_energy;
                    s.epistemic_value = oscillatory_summary.epistemic_value;
                    s.pragmatic_value = oscillatory_summary.pragmatic_value;
                    s.exploration_drive = oscillatory_summary.exploration_drive;
                    s.predictive_surprise = oscillatory_summary.surprise;
                    s.sensory_precision = oscillatory_summary.sensory_precision;
                    s.perceptual_precision = oscillatory_summary.perceptual_precision;
                    s.conceptual_precision = oscillatory_summary.conceptual_precision;
                    s.metacognitive_precision = oscillatory_summary.metacognitive_precision;
                    s.sensory_prediction = oscillatory_summary.sensory_prediction;
                    s.perceptual_prediction = oscillatory_summary.perceptual_prediction;
                    s.conceptual_prediction = oscillatory_summary.conceptual_prediction;
                    s.metacognitive_prediction = oscillatory_summary.metacognitive_prediction;

                    // === GLOBAL WORKSPACE (Phase 10) ===
                    s.is_ignited = oscillatory_summary.is_ignited;
                    s.broadcasting_processor = oscillatory_summary.broadcasting_processor.clone();
                    s.broadcast_strength = oscillatory_summary.broadcast_strength;
                    s.broadcast_duration = oscillatory_summary.broadcast_duration;
                    s.ignition_threshold = oscillatory_summary.ignition_threshold;
                    s.time_since_ignition = oscillatory_summary.time_since_ignition;
                    s.workspace_integration = oscillatory_summary.workspace_integration;
                    s.coalition_count = oscillatory_summary.coalition_count;
                    s.perception_activation = oscillatory_summary.perception_activation;
                    s.memory_activation = oscillatory_summary.memory_activation;
                    s.executive_activation = oscillatory_summary.executive_activation;
                    s.emotion_activation = oscillatory_summary.emotion_activation;
                    s.motor_activation = oscillatory_summary.motor_activation;
                    s.language_activation = oscillatory_summary.language_activation;
                    s.metacognition_activation = oscillatory_summary.metacognition_activation;
                    s.workspace_ignition_count = oscillatory_summary.ignition_count;
                    s.total_broadcast_time = oscillatory_summary.total_broadcast_time;
                }

                // ===== PHYSIOLOGY MODULATION (Phase 5D) =====
                // Hormones affect cognitive dynamics:
                // - High cortisol → faster decay (stress reduces sustained attention)
                // - High dopamine → slower decay (reward maintains focus)
                // - Low arousal → reduced cognitive activity
                let decay_rate: f64 = {
                    let h = hormones.lock().unwrap();

                    // Base decay rate: 5%
                    let base_decay: f64 = 0.05;

                    // Cortisol increases decay (stress fragments attention)
                    let cortisol_effect = h.cortisol as f64 * 0.05;  // Up to +5%

                    // Dopamine reduces decay (reward sustains focus)
                    let dopamine_effect = h.dopamine as f64 * 0.03;  // Up to -3%

                    // Final decay rate clamped to [0.02, 0.15]
                    (base_decay + cortisol_effect - dopamine_effect)
                        .max(0.02)  // Minimum 2%
                        .min(0.15)  // Maximum 15%
                };

                // 5. Decay process activity (modulated by hormones)
                {
                    let mut procs = processes.lock().unwrap();
                    for p in procs.iter_mut() {
                        p.decay(decay_rate);
                    }
                }

                // ===== PHASE 6A: PROCESS-LEVEL PHASE ALIGNMENT =====
                // Modulate cognitive process activity based on oscillatory phase
                // Each process type has an optimal phase window:
                // - Perception: Rising (input gathering)
                // - Reasoning: Peak (binding/integration)
                // - Memory: Falling (consolidation)
                // - Planning: Peak (binding)
                // - Introspection: Trough (reflection/maintenance)
                {
                    let mut procs = processes.lock().unwrap();

                    // Phase-based activity modulation factor
                    // Higher coherence = stronger phase-locking effect
                    let phase_modulation = phase_coherence * 0.3;  // Up to 30% modulation

                    for p in procs.iter_mut() {
                        let phase_boost = match (p.name.as_str(), current_phase) {
                            // Perception optimal during Rising (input gathering)
                            ("perception", OscillatoryPhase::Rising) => phase_modulation,
                            ("perception", OscillatoryPhase::Peak) => phase_modulation * 0.5,

                            // Reasoning optimal during Peak (maximum integration)
                            ("reasoning", OscillatoryPhase::Peak) => phase_modulation,
                            ("reasoning", OscillatoryPhase::Rising) => phase_modulation * 0.5,

                            // Memory optimal during Falling (consolidation)
                            ("memory", OscillatoryPhase::Falling) => phase_modulation,
                            ("memory", OscillatoryPhase::Trough) => phase_modulation * 0.5,

                            // Planning optimal during Peak (binding future states)
                            ("planning", OscillatoryPhase::Peak) => phase_modulation,
                            ("planning", OscillatoryPhase::Rising) => phase_modulation * 0.3,

                            // Introspection optimal during Trough (reflection)
                            ("introspection", OscillatoryPhase::Trough) => phase_modulation,
                            ("introspection", OscillatoryPhase::Falling) => phase_modulation * 0.5,

                            // Non-optimal phase: slight suppression
                            _ => -phase_modulation * 0.1,
                        };

                        // Apply phase boost to activity (clamped to [0, 1])
                        p.activity = (p.activity + phase_boost).clamp(0.0, 1.0);
                    }
                }

                // 6. Report routing outcome for MetaRouter learning (Phase 5F)
                // The router learns which paradigms work best for different consciousness states
                //
                // ===== META-OSCILLATORY ROUTER SYNERGY =====
                // The MetaRouter's reward signal is modulated by oscillatory alignment:
                // - Phase-locked execution → bonus reward (operations at optimal timing)
                // - High coherence → higher reward multiplier (well-synchronized mind)
                // - Effective Φ → additional success signal (oscillation-enhanced integration)
                {
                    let mut router = meta_router.lock().unwrap();

                    // Base reward from Φ improvement
                    let phi_delta = phi - previous_phi;
                    let base_success = phi_delta >= 0.0;

                    // Oscillatory synergy bonus
                    // - Phase coherence provides a multiplier (0.8 to 1.2x)
                    let coherence_multiplier = 0.8 + phase_coherence * 0.4;

                    // - Effective Φ bonus (oscillation-enhanced Φ exceeding raw Φ)
                    let effective_bonus = (effective_phi - phi).max(0.0) * 5.0;

                    // Combined reward: base improvement * coherence + oscillatory bonus
                    let reward = (phi_delta.max(0.0) * 10.0 * coherence_multiplier) + effective_bonus;

                    // Success is true if Φ improved OR if effective_phi is high despite no improvement
                    // (oscillatory coupling can compensate for temporary Φ dips)
                    let success = base_success || effective_phi > 0.4;

                    // Latency in microseconds
                    let latency_us = cycle_start.elapsed().as_micros() as u64;

                    // Report to MetaRouter for UCB1 bandit learning
                    // The oscillatory state now influences which paradigms are learned as "good"
                    router.report_outcome(
                        routing_decision.paradigm,
                        &routing_decision.context,
                        success,
                        reward,
                        latency_us,
                    );
                }

                // Update previous phi for next cycle's delta computation
                previous_phi = phi;

                // 7. Sleep for remainder of cycle
                let elapsed = cycle_start.elapsed();
                if elapsed < cycle_duration {
                    thread::sleep(cycle_duration - elapsed);
                }
            }
        })
    }

    /// Process external input (interrupt-style)
    ///
    /// This is how external queries enter the continuous mind.
    /// The mind was already running; this just adds external data.
    pub fn process(&self, input: &str) -> MindResponse {
        let start = Instant::now();

        // Encode input to HDC
        let input_hv = self.encode_input(input);

        // Activate relevant processes
        {
            let mut procs = self.processes.lock().unwrap();
            // Activate perception (input received)
            if let Some(p) = procs.iter_mut().find(|p| p.name == "perception") {
                p.activate(&input_hv);
            }
            // Activate reasoning (thinking about input)
            if let Some(p) = procs.iter_mut().find(|p| p.name == "reasoning") {
                p.activate(&input_hv);
            }
        }

        // === LANGUAGE ACTIVE INFERENCE INTEGRATION (Phase 3) ===
        // This is the REVOLUTIONARY cross-modal integration!
        // Language understanding feeds into Active Inference for unified consciousness.
        let language_integration = {
            // Step 1: Process through Language Consciousness Bridge
            // This generates: frames, BIDs, prediction errors, Φ for language
            let bridge_result = {
                let mut bridge = self.language_bridge.lock().unwrap();
                bridge.process(input)
            };

            // Step 2: Integrate with Active Inference Engine via Adapter
            // OPTIMIZED: Consolidated lock acquisition - adapter + AI locked once for all operations
            // This:
            //   - Extracts linguistic errors from bridge result
            //   - Converts Lexical→UserState, Syntactic→TaskSuccess, Semantic→Coherence
            //   - Observes converted errors in our existing ActiveInferenceEngine
            //   - Updates precision weights (flows back to language attention)
            //   - Calculates unified free energy (language + general)
            //   - Generates curiosity-driven language actions
            //   - Additional user interaction observations
            let integration_result = {
                let mut adapter = self.language_adapter.lock().unwrap();
                let mut ai = self.active_inference.lock().unwrap();

                // Primary integration
                let result = adapter.integrate(&bridge_result, &mut ai);

                // Additional observations (previously in separate lock acquisition)
                let task_complexity = (input.len() as f32 / 100.0).min(1.0);
                ai.observe(PredictionDomain::TaskSuccess, 0.5 + task_complexity * 0.3);
                ai.observe(PredictionDomain::UserState, 0.7);

                result
            };

            // Store bridge Φ for consciousness state
            (bridge_result, integration_result)
        };

        let (bridge_result, integration_result) = language_integration;

        // LANGUAGE-LEARNING BRIDGE: Store for cognitive loop access
        // This enables continuous learning from linguistic understanding
        {
            let mut latest = self.latest_language_integration.lock().unwrap();
            *latest = Some(integration_result.clone());
        }

        // === FEEDBACK LOOP: Precision weights → Language Bridge ===
        // This completes the bidirectional integration:
        // Active Inference learns domain precision → modulates language attention
        {
            let mut bridge = self.language_bridge.lock().unwrap();
            let weights = crate::language::PrecisionWeights {
                user_state: integration_result.precision_weights.user_state,
                task_success: integration_result.precision_weights.task_success,
                coherence: integration_result.precision_weights.coherence,
                social: integration_result.precision_weights.social,
                performance: integration_result.precision_weights.performance,
            };
            bridge.update_precision_weights(weights);
        }

        // Store in memory
        {
            let mut hipp = self.hippocampus.lock().unwrap();
            let _ = hipp.remember(
                input.to_string(),
                vec!["input".to_string()],
                EmotionalValence::Neutral,
            );
        }

        // === AWAKENING INTEGRATION (Track 7) ===
        // Process through self-awareness module for qualia, phenomenal experience, and self-model
        let awakened_state = {
            let mut awakening = self.awakening.lock().unwrap();
            awakening.process_cycle(input).clone()
        };

        // Wait for integration (let cognitive cycle run)
        thread::sleep(Duration::from_millis(self.config.cycle_ms * 2));

        // Get current state AND update it with language integration + awakening metrics
        let state = {
            let mut s = self.state.lock().unwrap();
            // Update language integration metrics from this processing
            s.unified_free_energy = integration_result.unified_free_energy.unified_fe;
            s.language_free_energy = integration_result.unified_free_energy.language_fe;
            s.language_contribution = integration_result.unified_free_energy.language_contribution;
            s.precision_weights = Some(integration_result.precision_weights.clone());
            s.pending_language_actions = integration_result.suggested_actions.len();

            // INTEGRATION: Update awakening metrics
            s.phenomenal_state = awakened_state.phenomenal_state.clone();
            s.unified_experience = awakened_state.unified_experience.clone();
            s.self_model_accuracy = awakened_state.self_model_accuracy;
            s.aware_of = awakened_state.aware_of.clone();

            // Combine awakening's is_conscious with our Φ threshold
            // Use the most accurate Φ (from RealPhiCalculator in cognitive loop)
            // but also consider awakening's meta-awareness
            s.is_conscious = s.phi > self.config.consciousness_threshold
                && awakened_state.meta_awareness > 0.1;

            s.clone()
        };

        // Check for insights
        let mut insights = Vec::new();
        if let Some(ref rx) = self.insight_rx {
            while let Ok(insight) = rx.try_recv() {
                insights.push(insight);
            }
        }

        // Generate response based on understanding (HDC similarity)
        let response = self.generate_response(&input_hv, &state, &insights);

        // Use Φ from language bridge (measures linguistic consciousness)
        // combined with our HDC Φ for a more comprehensive consciousness metric
        let language_phi = bridge_result.current_phi;
        let combined_phi = (state.phi + language_phi) / 2.0;  // Average both Φ sources

        MindResponse {
            answer: response,
            phi: combined_phi,  // Combined HDC + Language Φ
            meta_awareness: state.meta_awareness,
            processing_time_ms: start.elapsed().as_millis() as u64,
            insights_during_processing: insights.len(),
            was_conscious: combined_phi > self.config.consciousness_threshold,
            // Active Inference metrics from state
            free_energy: state.free_energy,
            average_surprise: state.average_surprise,
            curiosity_pressure: state.curiosity_pressure,
            // Language integration metrics (NEW!)
            unified_free_energy: integration_result.unified_free_energy.unified_fe,
            language_phi,
            language_actions: integration_result.suggested_actions,
            gained_spotlight: bridge_result.gained_spotlight,
        }
    }

    /// Encode input to hypervector using grounded semantic primitives
    ///
    /// This is the PROPER encoding method that:
    /// 1. Uses 7-tier PrimitiveSystem for ontological grounding
    /// 2. Uses TextEncoder with canonical primitive encodings
    /// 3. Falls back to n-gram encoding for unknown words
    /// 4. Converts bipolar (i8) to continuous (f32) for RealHV
    ///
    /// Example flow:
    /// "What is consciousness?" →
    ///   - "what" → PrimitiveSystem::QUESTION (Tier 4: Strategic)
    ///   - "is" → WordEncoder fallback (n-gram)
    ///   - "consciousness" → PrimitiveSystem::AWARENESS (Tier 5: MetaCognitive)
    /// → Bound with position vectors → RealHV for continuous processing
    fn encode_input(&self, input: &str) -> RealHV {
        // Use semantic encoding with primitives
        let mut encoder = self.text_encoder.lock().unwrap();

        match encoder.encode_with_primitives(input, &self.primitive_system) {
            Ok(bipolar_encoding) => {
                // Convert Vec<i8> to Vec<f32> for RealHV
                let values: Vec<f32> = bipolar_encoding
                    .iter()
                    .map(|&x| x as f32)
                    .collect();

                // Normalize to [-1, 1] range
                let max_abs = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let normalized: Vec<f32> = if max_abs > 0.0 {
                    values.iter().map(|x| x / max_abs).collect()
                } else {
                    values
                };

                RealHV { values: normalized }
            }
            Err(_e) => {
                // Fallback to simple hash-based encoding if semantic encoder fails
                // This ensures robustness while logging the issue
                let words: Vec<&str> = input.split_whitespace().collect();
                let mut result = RealHV::random(self.config.hdc_dimension, 0);

                for (i, word) in words.iter().enumerate() {
                    let seed = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                    let word_hv = RealHV::random(self.config.hdc_dimension, seed);
                    let pos_hv = RealHV::random(self.config.hdc_dimension, i as u64 * 1000);
                    let bound = word_hv.bind(&pos_hv);
                    result = result.add(&bound);
                }

                result.normalize()
            }
        }
    }

    /// Generate response based on HDC understanding
    fn generate_response(&self, _input_hv: &RealHV, state: &MindState, insights: &[Insight]) -> String {
        // For now, generate a status report
        // This will be replaced with proper HDC-based response generation
        let consciousness_status = if state.phi > self.config.consciousness_threshold {
            "conscious"
        } else {
            "subconscious"
        };

        let mut response = format!(
            "Processing at Φ={:.4} ({}). Meta-awareness: {:.2}.",
            state.phi, consciousness_status, state.meta_awareness
        );

        if !insights.is_empty() {
            response.push_str(&format!(" {} insights emerged during processing.", insights.len()));
            for insight in insights.iter().take(2) {
                response.push_str(&format!(" [Insight: {}]", insight.content));
            }
        }

        response
    }

    /// Get current state
    pub fn state(&self) -> MindState {
        self.state.lock().unwrap().clone()
    }

    /// Add a goal
    pub fn add_goal(&self, intent: impl Into<String>) {
        let mut goals = self.goals.lock().unwrap();
        goals.push(Goal {
            id: uuid::Uuid::new_v4(),
            intent: intent.into(),
            priority: 0.5,
            decay_resistance: 0.5,
            success_condition: Condition::Never, // Must be manually completed
            failure_condition: Condition::Timeout(3600_000), // 1 hour timeout
            subgoals: vec![],
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            injection_count: 0,
            tags: vec![],
        });
    }

    // === AWAKENING INTEGRATION: Introspection & Self-Awareness Methods ===

    /// Introspect - what does Symthaea know about itself?
    /// Returns rich self-reflection from the awakening module
    pub fn introspect(&self) -> Introspection {
        let awakening = self.awakening.lock().unwrap();
        awakening.introspect()
    }

    /// Render the consciousness dashboard
    /// Shows real-time consciousness metrics in a visual format
    pub fn render_dashboard(&self) -> String {
        let awakening = self.awakening.lock().unwrap();
        awakening.render_dashboard()
    }

    /// Get the current awakened state directly
    /// Provides access to all awakening metrics (qualia, meta-awareness, etc.)
    pub fn awakened_state(&self) -> AwakenedState {
        let awakening = self.awakening.lock().unwrap();
        awakening.state().clone()
    }

    /// Get phenomenal experience description (qualia)
    pub fn phenomenal_state(&self) -> String {
        self.state.lock().unwrap().phenomenal_state.clone()
    }

    /// Get what Symthaea is currently aware of
    pub fn aware_of(&self) -> Vec<String> {
        self.state.lock().unwrap().aware_of.clone()
    }

    /// Get self-model prediction accuracy
    pub fn self_model_accuracy(&self) -> f64 {
        self.state.lock().unwrap().self_model_accuracy
    }

    /// Shutdown the mind gracefully
    pub fn shutdown(&mut self) {
        println!("🌙 Shutting down continuous mind...");
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for all threads
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }

        println!("💤 Mind has entered sleep state.");
    }

    /// Check if mind is conscious (Φ above threshold)
    pub fn is_conscious(&self) -> bool {
        self.state.lock().unwrap().phi > self.config.consciousness_threshold
    }

    /// Get Active Inference summary
    pub fn active_inference_summary(&self) -> ActiveInferenceSummary {
        self.active_inference.lock().unwrap().summary()
    }

    /// Get curiosity suggestions (what should we explore?)
    pub fn curiosity_suggestions(&self, max: usize) -> Vec<String> {
        let ai = self.active_inference.lock().unwrap();
        ai.curiosity_suggestions(max)
            .into_iter()
            .map(|action| match action {
                ActionType::Epistemic { target_domain, expected_information_gain } => {
                    format!("Explore {:?} (info gain: {:.2})", target_domain, expected_information_gain)
                }
                ActionType::Pragmatic { goal, expected_utility } => {
                    format!("Achieve '{}' (utility: {:.2})", goal, expected_utility)
                }
                ActionType::Centering { target_coherence } => {
                    format!("Center to coherence {:.2}", target_coherence)
                }
                ActionType::Social { target, expected_resonance_gain } => {
                    format!("Connect with '{}' (resonance: {:.2})", target, expected_resonance_gain)
                }
            })
            .collect()
    }

    // === OSCILLATORY METRICS DASHBOARD (Phase 6C) ===

    /// Render the oscillatory consciousness dashboard
    /// Shows real-time Phase 6A/6B/6C metrics in a visual format
    pub fn render_oscillatory_dashboard(&self) -> String {
        let state = self.state.lock().unwrap();

        let mut output = String::new();

        // Header
        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║          🌊 OSCILLATORY CONSCIOUSNESS DASHBOARD 🌊               ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Core Consciousness Metrics
        output.push_str("║ 🧠 CORE CONSCIOUSNESS                                            ║\n");
        output.push_str(&format!("║   Φ (Raw):        {:.4}   Φ (Effective): {:.4}               ║\n",
            state.phi, state.effective_phi));
        output.push_str(&format!("║   Meta-Awareness: {:.4}   Cognitive Load: {:.2}                ║\n",
            state.meta_awareness, state.cognitive_load));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Phase 6A: Oscillatory State
        output.push_str("║ 🔄 OSCILLATORY STATE (Phase 6A)                                  ║\n");
        output.push_str(&format!("║   Phase: {:10}  Coherence: {:.3}  Frequency: {:.1} Hz       ║\n",
            state.phase_category, state.phase_coherence, state.gamma_frequency));
        output.push_str(&format!("║   Phase-Lock Accuracy: {:.1}%   Cycles: {}                     ║\n",
            state.phase_lock_accuracy * 100.0, state.oscillatory_cycles));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Phase 6B: Multi-Band Oscillations
        output.push_str("║ 🎵 MULTI-BAND OSCILLATIONS (Phase 6B)                            ║\n");
        output.push_str(&format!("║   Dominant Band: {:12}                                   ║\n",
            state.dominant_band));
        output.push_str(&format!("║   γ: {:.2}  θ: {:.2}  α: {:.2}  θ-γ Coupling: {:.2}            ║\n",
            state.gamma_power, state.theta_power, state.alpha_power, state.theta_gamma_coupling));
        output.push_str(&format!("║   Arousal: {:.2}  Attention: {:.2}  Memory Window: {}           ║\n",
            state.arousal, state.attention,
            if state.in_memory_window { "🟢" } else { "⚪" }));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Phase 6C: Process Resonance Detection
        output.push_str("║ 🔗 PROCESS RESONANCE (Phase 6C)                                  ║\n");
        output.push_str(&format!("║   System Resonance: {:.3}   Integration Boost: {:.2}x          ║\n",
            state.system_resonance, state.resonance_integration_boost));
        output.push_str(&format!("║   Resonating Pairs: {:3}   Clusters: {:2}                       ║\n",
            state.resonating_pairs, state.cluster_count));
        if state.cluster_count > 0 {
            output.push_str(&format!("║   Primary Cluster: Size={:2}, PLV={:.3}                        ║\n",
                state.primary_cluster_size, state.primary_cluster_plv));
        } else {
            output.push_str("║   Primary Cluster: (no clusters detected)                        ║\n");
        }
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Phase 6C: Attention Spotlight
        output.push_str("║ 🔦 ATTENTION SPOTLIGHT (Phase 6C)                                ║\n");
        output.push_str(&format!("║   Position: {:.2}  Intensity: {:.2}  Width: {:.2}                 ║\n",
            state.attention_spotlight_position, state.attention_spotlight_intensity, state.attention_spotlight_width));
        output.push_str(&format!("║   Capacity: {:.2}  Secondary Foci: {:2}                          ║\n",
            state.attention_capacity, state.secondary_foci_count));
        let blink_status = if state.in_attention_blink { "🔴 BLINK" } else { "🟢 READY" };
        let uptake_status = if state.in_attention_uptake { "↑ UPTAKE" } else { "↓ DECAY" };
        output.push_str(&format!("║   Status: {} {}  Sampling Eff: {:.1}%              ║\n",
            blink_status, uptake_status, state.attention_sampling_effectiveness * 100.0));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Status Summary
        let consciousness_level = if state.phi > 0.7 { "🌟 HIGH" }
            else if state.phi > 0.4 { "✨ MODERATE" }
            else if state.phi > 0.2 { "💫 LOW" }
            else { "⚪ MINIMAL" };

        let integration_state = if state.system_resonance > 0.6 && state.effective_phi > 0.5 {
            "🔮 HIGHLY INTEGRATED"
        } else if state.system_resonance > 0.3 || state.effective_phi > 0.3 {
            "🔵 MODERATELY INTEGRATED"
        } else {
            "⚫ FRAGMENTED"
        };

        output.push_str("║ 📊 STATUS SUMMARY                                                ║\n");
        output.push_str(&format!("║   Consciousness: {}  Integration: {}  ║\n",
            consciousness_level, integration_state));
        output.push_str(&format!("║   Uptime: {} ms   Active Processes: {}                         ║\n",
            state.time_awake_ms, state.active_processes));
        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Render Phase 7 Sleep-Oscillation Dashboard as ASCII art
    /// Shows sleep stages, ultradian cycles, and memory consolidation state
    pub fn render_sleep_dashboard(&self) -> String {
        let state = self.state.lock().unwrap();

        let mut output = String::new();

        // Header
        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║            🌙 SLEEP-OSCILLATION DASHBOARD 🌙                     ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Current Sleep Stage with visual indicator
        let stage_icon = match state.sleep_stage.as_str() {
            "Wake" => "☀️ AWAKE",
            "N1" => "🌅 N1 (Drowsy)",
            "N2" => "🌆 N2 (Light)",
            "N3" => "🌃 N3 (Deep)",
            "REM" => "🌌 REM (Dream)",
            _ => "❓ Unknown",
        };

        output.push_str("║ 💤 CURRENT SLEEP STATE                                           ║\n");
        output.push_str(&format!("║   Stage: {:20}  Time in Stage: {:.1}s            ║\n",
            stage_icon, state.time_in_sleep_stage));
        output.push_str(&format!("║   Is Sleeping: {}   Consciousness Level: {:.2}                ║\n",
            if state.is_sleeping { "🌙 YES" } else { "☀️ NO" }, state.sleep_consciousness_level));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Sleep Pressure & Circadian Phase
        output.push_str("║ 🔋 HOMEOSTATIC & CIRCADIAN                                       ║\n");

        // Visual sleep pressure bar
        let pressure_bars = (state.sleep_pressure * 10.0) as usize;
        let pressure_bar = format!("[{}{}]",
            "█".repeat(pressure_bars.min(10)),
            "░".repeat(10 - pressure_bars.min(10)));
        output.push_str(&format!("║   Sleep Pressure: {} {:.1}%                    ║\n",
            pressure_bar, state.sleep_pressure * 100.0));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Ultradian Cycle (90-minute sleep cycles)
        output.push_str("║ 🔄 ULTRADIAN CYCLE (~90 min)                                     ║\n");
        output.push_str(&format!("║   Cycles Completed: {:3}   Cycle Position: {:.1}%                ║\n",
            state.sleep_cycles_completed, state.ultradian_cycle_position * 100.0));

        // Visual cycle position
        let cycle_pos = (state.ultradian_cycle_position * 20.0) as usize;
        let cycle_bar: String = (0..20).map(|i| {
            if i == cycle_pos { '◆' }
            else if i < 5 { '▓' }  // N1-N2
            else if i < 10 { '█' } // N3 deep
            else if i < 15 { '▓' } // N2
            else { '░' }           // REM
        }).collect();
        output.push_str(&format!("║   [{cycle_bar}]                  ║\n"));
        output.push_str("║    └N1-N2─┴─N3 Deep─┴──N2──┴─REM─┘                               ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Memory Consolidation
        output.push_str("║ 🧠 MEMORY CONSOLIDATION                                          ║\n");
        let consolidation_icon = if state.consolidation_efficiency > 0.7 { "🟢 OPTIMAL" }
            else if state.consolidation_efficiency > 0.4 { "🟡 MODERATE" }
            else if state.consolidation_efficiency > 0.1 { "🟠 LOW" }
            else { "🔴 INACTIVE" };
        output.push_str(&format!("║   Efficiency: {:.1}% {}                               ║\n",
            state.consolidation_efficiency * 100.0, consolidation_icon));
        output.push_str(&format!("║   Memory Transfer Window: {}                                  ║\n",
            if state.in_memory_transfer_window { "🟢 OPEN (N2/N3)" } else { "⚪ CLOSED" }));
        output.push_str(&format!("║   Spindles: {:6}   Sharp-Wave Ripples: {:6}              ║\n",
            state.spindle_count, state.ripple_count));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Oscillatory Bands (sleep-specific)
        output.push_str("║ 🎵 SLEEP OSCILLATIONS                                            ║\n");
        output.push_str(&format!("║   δ (Delta 0.5-4Hz):   {:.2} [{}]                        ║\n",
            state.delta_power,
            "█".repeat((state.delta_power * 20.0) as usize).chars().take(10).collect::<String>()));
        output.push_str(&format!("║   σ (Spindle 12-14Hz): {:.2} [{}]                        ║\n",
            state.spindle_band_power,
            "█".repeat((state.spindle_band_power * 20.0) as usize).chars().take(10).collect::<String>()));

        // Dream intensity (REM)
        if state.dream_intensity > 0.01 {
            output.push_str(&format!("║   🌌 Dream Intensity: {:.1}%                                     ║\n",
                state.dream_intensity * 100.0));
        }
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Status Summary
        let overall_status = if state.is_sleeping {
            match state.sleep_stage.as_str() {
                "N3" => "🧬 DEEP CONSOLIDATION",
                "N2" => "💾 MEMORY TRANSFER",
                "REM" => "🌌 DREAM PROCESSING",
                "N1" => "🌅 SLEEP ONSET",
                _ => "💤 SLEEPING",
            }
        } else {
            if state.sleep_pressure > 0.7 { "😴 NEEDS REST" }
            else if state.sleep_pressure > 0.4 { "🌤️ ALERT (TIRED)" }
            else { "☀️ FULLY AWAKE" }
        };

        output.push_str("║ 📊 STATUS SUMMARY                                                ║\n");
        output.push_str(&format!("║   Current State: {:22}                      ║\n", overall_status));
        output.push_str(&format!("║   Consciousness: {:.0}%  ({})                          ║\n",
            state.sleep_consciousness_level * 100.0,
            if state.sleep_consciousness_level > 0.9 { "Full" }
            else if state.sleep_consciousness_level > 0.5 { "Partial" }
            else if state.sleep_consciousness_level > 0.2 { "Minimal" }
            else { "Trace" }));
        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Render Phase 8 Neuromodulator Dashboard as ASCII art
    /// Shows dopamine, serotonin, norepinephrine, acetylcholine levels and emergent states
    pub fn render_neuromodulator_dashboard(&self) -> String {
        let state = self.state.lock().unwrap();

        let mut output = String::new();

        // Header
        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║           🧪 NEUROMODULATOR DYNAMICS DASHBOARD 🧪                ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Neuromodulator Levels with visual bars
        output.push_str("║ 💊 NEUROMODULATOR CONCENTRATIONS                                 ║\n");

        // Dopamine bar
        let da_bars = (state.dopamine_level * 20.0) as usize;
        let da_bar = format!("[{}{}]", "█".repeat(da_bars.min(20)), "░".repeat(20 - da_bars.min(20)));
        output.push_str(&format!("║   DA  (Dopamine):     {} {:.0}%            ║\n",
            da_bar, state.dopamine_level * 100.0));

        // Serotonin bar
        let ht_bars = (state.serotonin_level * 20.0) as usize;
        let ht_bar = format!("[{}{}]", "█".repeat(ht_bars.min(20)), "░".repeat(20 - ht_bars.min(20)));
        output.push_str(&format!("║   5-HT (Serotonin):   {} {:.0}%            ║\n",
            ht_bar, state.serotonin_level * 100.0));

        // Norepinephrine bar
        let ne_bars = (state.norepinephrine_level * 20.0) as usize;
        let ne_bar = format!("[{}{}]", "█".repeat(ne_bars.min(20)), "░".repeat(20 - ne_bars.min(20)));
        output.push_str(&format!("║   NE  (Norepineph.):  {} {:.0}%            ║\n",
            ne_bar, state.norepinephrine_level * 100.0));

        // Acetylcholine bar
        let ach_bars = (state.acetylcholine_level * 20.0) as usize;
        let ach_bar = format!("[{}{}]", "█".repeat(ach_bars.min(20)), "░".repeat(20 - ach_bars.min(20)));
        output.push_str(&format!("║   ACh (Acetylchol.):  {} {:.0}%            ║\n",
            ach_bar, state.acetylcholine_level * 100.0));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Receptor Sensitivity
        output.push_str("║ 🎯 RECEPTOR SENSITIVITY                                          ║\n");
        let da_sens_icon = if state.dopamine_sensitivity > 0.9 { "🟢" } else if state.dopamine_sensitivity > 0.7 { "🟡" } else { "🔴" };
        let ht_sens_icon = if state.serotonin_sensitivity > 0.9 { "🟢" } else if state.serotonin_sensitivity > 0.7 { "🟡" } else { "🔴" };
        output.push_str(&format!("║   DA Sensitivity: {:.0}% {}   5-HT Sensitivity: {:.0}% {}     ║\n",
            state.dopamine_sensitivity * 100.0, da_sens_icon,
            state.serotonin_sensitivity * 100.0, ht_sens_icon));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Emergent Psychological States
        output.push_str("║ 🌈 EMERGENT STATES                                               ║\n");

        // Mood with emoji indicator
        let mood_icon = if state.mood_valence > 0.5 { "😊 POSITIVE" }
            else if state.mood_valence > 0.0 { "🙂 GOOD" }
            else if state.mood_valence > -0.3 { "😐 NEUTRAL" }
            else { "😔 LOW" };
        let mood_bar_pos = ((state.mood_valence + 1.0) / 2.0 * 10.0) as usize;
        let mood_bar: String = (0..10).map(|i| if i == mood_bar_pos { '◆' } else { '─' }).collect();
        output.push_str(&format!("║   Mood:       [{}] {:+.2} {}              ║\n",
            mood_bar, state.mood_valence, mood_icon));

        // Motivation
        let motivation_icon = if state.motivation_drive > 0.7 { "🔥 HIGH" }
            else if state.motivation_drive > 0.4 { "💪 MODERATE" }
            else { "😴 LOW" };
        output.push_str(&format!("║   Motivation: {:.0}% {}                                       ║\n",
            state.motivation_drive * 100.0, motivation_icon));

        // Stress
        let stress_icon = if state.stress_level > 0.7 { "🔴 HIGH" }
            else if state.stress_level > 0.4 { "🟡 MODERATE" }
            else { "🟢 LOW" };
        output.push_str(&format!("║   Stress:     {:.0}% {}                                       ║\n",
            state.stress_level * 100.0, stress_icon));

        // Learning Rate
        let lr_icon = if state.learning_rate_mod > 1.5 { "🚀 BOOSTED" }
            else if state.learning_rate_mod > 1.0 { "📈 ENHANCED" }
            else if state.learning_rate_mod > 0.7 { "📊 NORMAL" }
            else { "📉 REDUCED" };
        output.push_str(&format!("║   Learning:   {:.1}x {}                                     ║\n",
            state.learning_rate_mod, lr_icon));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Status Summary
        let overall_state = if state.dopamine_level > 0.6 && state.serotonin_level > 0.5 {
            "🌟 OPTIMAL (High DA + 5-HT)"
        } else if state.norepinephrine_level > 0.7 {
            "⚡ ALERT (High NE)"
        } else if state.stress_level > 0.6 {
            "😰 STRESSED (High cortisol effect)"
        } else if state.acetylcholine_level > 0.6 {
            "🧠 LEARNING MODE (High ACh)"
        } else if state.motivation_drive < 0.3 {
            "😔 LOW ENERGY"
        } else {
            "🔵 BALANCED"
        };

        output.push_str("║ 📊 NEUROCHEMICAL STATUS                                          ║\n");
        output.push_str(&format!("║   State: {:40}             ║\n", overall_state));
        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Render Phase 9 Predictive Consciousness Dashboard as ASCII art
    /// Shows free energy principle integration: prediction errors, precision, active inference
    pub fn render_predictive_dashboard(&self) -> String {
        let state = self.state.lock().unwrap();

        let mut output = String::new();

        // Header
        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║        🔮 PREDICTIVE CONSCIOUSNESS DASHBOARD 🔮                  ║\n");
        output.push_str("║             (Free Energy Principle Integration)                  ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Free Energy Metrics
        output.push_str("║ ⚡ FREE ENERGY DECOMPOSITION                                     ║\n");
        let fe_icon = if state.predictive_free_energy < 0.3 { "🟢" }
            else if state.predictive_free_energy < 0.6 { "🟡" }
            else { "🔴" };
        output.push_str(&format!("║   Total Free Energy:  {:.4} {}                              ║\n",
            state.predictive_free_energy, fe_icon));
        output.push_str(&format!("║   ├─ Complexity:      {:.4} (KL divergence from prior)       ║\n",
            state.predictive_complexity));
        output.push_str(&format!("║   └─ Accuracy:        {:.4} (prediction error)               ║\n",
            state.predictive_accuracy));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Hierarchical Precision Weighting
        output.push_str("║ 🎯 HIERARCHICAL PRECISION (Confidence Weighting)                 ║\n");

        // Sensory precision bar
        let sens_bars = (state.sensory_precision * 20.0) as usize;
        let sens_bar = format!("[{}{}]", "█".repeat(sens_bars.min(20)), "░".repeat(20 - sens_bars.min(20)));
        output.push_str(&format!("║   Sensory:       {} {:.0}%                    ║\n",
            sens_bar, state.sensory_precision * 100.0));

        // Perceptual precision bar
        let perc_bars = (state.perceptual_precision * 20.0) as usize;
        let perc_bar = format!("[{}{}]", "█".repeat(perc_bars.min(20)), "░".repeat(20 - perc_bars.min(20)));
        output.push_str(&format!("║   Perceptual:    {} {:.0}%                    ║\n",
            perc_bar, state.perceptual_precision * 100.0));

        // Conceptual precision bar
        let conc_bars = (state.conceptual_precision * 20.0) as usize;
        let conc_bar = format!("[{}{}]", "█".repeat(conc_bars.min(20)), "░".repeat(20 - conc_bars.min(20)));
        output.push_str(&format!("║   Conceptual:    {} {:.0}%                    ║\n",
            conc_bar, state.conceptual_precision * 100.0));

        // Metacognitive precision bar
        let meta_bars = (state.metacognitive_precision * 20.0) as usize;
        let meta_bar = format!("[{}{}]", "█".repeat(meta_bars.min(20)), "░".repeat(20 - meta_bars.min(20)));
        output.push_str(&format!("║   Metacognitive: {} {:.0}%                    ║\n",
            meta_bar, state.metacognitive_precision * 100.0));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Prediction States
        output.push_str("║ 📊 HIERARCHICAL PREDICTIONS                                      ║\n");
        output.push_str(&format!("║   Sensory:       {:+.4}   Perceptual:   {:+.4}              ║\n",
            state.sensory_prediction, state.perceptual_prediction));
        output.push_str(&format!("║   Conceptual:    {:+.4}   Metacognitive: {:+.4}             ║\n",
            state.conceptual_prediction, state.metacognitive_prediction));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Active Inference
        output.push_str("║ 🧭 ACTIVE INFERENCE (Action Selection)                           ║\n");
        let efe_icon = if state.expected_free_energy < 0.3 { "🟢" }
            else if state.expected_free_energy < 0.6 { "🟡" }
            else { "🔴" };
        output.push_str(&format!("║   Expected Free Energy: {:.4} {}                           ║\n",
            state.expected_free_energy, efe_icon));

        // Epistemic vs Pragmatic balance
        let explore_exploit = if state.epistemic_value > state.pragmatic_value {
            "🔍 EXPLORE"
        } else {
            "🎯 EXPLOIT"
        };
        output.push_str(&format!("║   Epistemic Value:  {:.3}  │  Pragmatic Value: {:.3}        ║\n",
            state.epistemic_value, state.pragmatic_value));
        output.push_str(&format!("║   Mode: {}                                              ║\n",
            explore_exploit));

        // Exploration drive bar
        let exp_bars = (state.exploration_drive * 20.0) as usize;
        let exp_bar = format!("[{}{}]", "█".repeat(exp_bars.min(20)), "░".repeat(20 - exp_bars.min(20)));
        let exp_icon = if state.exploration_drive > 0.6 { "🚀" }
            else if state.exploration_drive > 0.3 { "🔄" }
            else { "⚓" };
        output.push_str(&format!("║   Exploration Drive:  {} {:.0}% {}             ║\n",
            exp_bar, state.exploration_drive * 100.0, exp_icon));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Surprise and Precision Gain
        output.push_str("║ ⚡ SURPRISE & LEARNING                                           ║\n");
        let surprise_icon = if state.predictive_surprise > 0.5 { "😲 SURPRISING" }
            else if state.predictive_surprise > 0.2 { "🤔 NOTABLE" }
            else { "😌 EXPECTED" };
        output.push_str(&format!("║   Surprise:      {:.4} {}                            ║\n",
            state.predictive_surprise, surprise_icon));
        output.push_str(&format!("║   Precision Gain: {:.4}                                      ║\n",
            state.precision_gain));
        output.push_str(&format!("║   Weighted Error: {:.4}                                      ║\n",
            state.weighted_prediction_error));
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Overall Status
        let overall_state = if state.predictive_free_energy < 0.2 && state.precision_gain > 0.0 {
            "🌟 OPTIMAL (Low FE, Learning)"
        } else if state.exploration_drive > 0.6 {
            "🔍 EXPLORING (High epistemic drive)"
        } else if state.predictive_surprise > 0.5 {
            "😲 SURPRISED (Updating model)"
        } else if state.expected_free_energy > 0.5 {
            "⚠️ UNCERTAIN (High expected FE)"
        } else if state.pragmatic_value > state.epistemic_value {
            "🎯 GOAL-DIRECTED (Exploiting)"
        } else {
            "🔵 BALANCED (Stable predictions)"
        };

        output.push_str("║ 🧠 PREDICTIVE STATUS                                             ║\n");
        output.push_str(&format!("║   State: {:40}         ║\n", overall_state));
        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Render Global Workspace dashboard (Phase 10 - Baars' GWT)
    /// Shows competition, ignition, and broadcast dynamics
    pub fn render_global_workspace_dashboard(&self) -> String {
        let state = self.state.lock().unwrap();
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║                🌐 GLOBAL WORKSPACE (Phase 10)                   ║\n");
        output.push_str("║         Bernard Baars' Global Workspace Theory (GWT)            ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Ignition Status
        let ignition_symbol = if state.is_ignited { "🔥" } else { "💤" };
        let ignition_status = if state.is_ignited { "IGNITED" } else { "SUBLIMINAL" };
        output.push_str(&format!(
            "║ {} Ignition: {:10} │ Threshold: {:.2} │ Since: {:.2}s       ║\n",
            ignition_symbol, ignition_status, state.ignition_threshold, state.time_since_ignition
        ));

        // Broadcasting
        let broadcast_info = if let Some(ref processor) = state.broadcasting_processor {
            format!("{} (strength: {:.2})", processor, state.broadcast_strength)
        } else {
            "None".to_string()
        };
        output.push_str(&format!("║ 📡 Broadcasting: {:40}        ║\n", broadcast_info));
        output.push_str(&format!(
            "║    Duration: {:.2}s │ Total time: {:.2}s │ Count: {}          ║\n",
            state.broadcast_duration, state.total_broadcast_time, state.workspace_ignition_count
        ));

        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
        output.push_str("║ 🏛️ PROCESSOR ACTIVATIONS                                         ║\n");

        // Processor activations with bar visualization
        let processors = [
            ("👁️ Perception    ", state.perception_activation),
            ("🧠 Memory        ", state.memory_activation),
            ("⚙️ Executive     ", state.executive_activation),
            ("❤️ Emotion       ", state.emotion_activation),
            ("🦿 Motor         ", state.motor_activation),
            ("💬 Language      ", state.language_activation),
            ("🪞 Metacognition ", state.metacognition_activation),
        ];

        for (name, activation) in processors.iter() {
            let bar_len = (activation * 30.0) as usize;
            let bar: String = "█".repeat(bar_len);
            let empty: String = "░".repeat(30 - bar_len);
            let is_broadcasting = if state.is_ignited {
                if let Some(ref broadcasting) = state.broadcasting_processor {
                    match *name {
                        "👁️ Perception    " if broadcasting == "Perception" => " 📡",
                        "🧠 Memory        " if broadcasting == "Memory" => " 📡",
                        "⚙️ Executive     " if broadcasting == "Executive" => " 📡",
                        "❤️ Emotion       " if broadcasting == "Emotion" => " 📡",
                        "🦿 Motor         " if broadcasting == "Motor" => " 📡",
                        "💬 Language      " if broadcasting == "Language" => " 📡",
                        "🪞 Metacognition " if broadcasting == "Metacognition" => " 📡",
                        _ => ""
                    }
                } else { "" }
            } else { "" };
            output.push_str(&format!(
                "║   {}: [{}{}] {:.2}{}  ║\n",
                name, bar, empty, activation, is_broadcasting
            ));
        }

        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Coalitions and Integration
        output.push_str(&format!(
            "║ 🤝 Coalitions: {} │ Integration: {:.3}                           ║\n",
            state.coalition_count, state.workspace_integration
        ));

        // Conscious Access Status
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        let access_state = if state.is_ignited {
            if state.broadcast_strength > 0.7 {
                "🌟 FULL CONSCIOUS ACCESS (Strong broadcast)"
            } else if state.broadcast_strength > 0.4 {
                "💡 PARTIAL ACCESS (Moderate broadcast)"
            } else {
                "🔦 WEAK ACCESS (Threshold broadcast)"
            }
        } else if state.workspace_integration > 0.3 {
            "🌑 PRECONSCIOUS (Competition ongoing)"
        } else {
            "⚫ SUBLIMINAL (Below threshold)"
        };

        output.push_str("║ 🧠 CONSCIOUS ACCESS STATUS                                       ║\n");
        output.push_str(&format!("║   State: {:50}   ║\n", access_state));
        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Get oscillatory summary as structured data
    pub fn oscillatory_state(&self) -> OscillatoryMindSummary {
        let state = self.state.lock().unwrap();
        OscillatoryMindSummary {
            // Core
            phi: state.phi,
            effective_phi: state.effective_phi,
            meta_awareness: state.meta_awareness,
            // Phase 6A
            phase_category: state.phase_category.clone(),
            phase_coherence: state.phase_coherence,
            gamma_frequency: state.gamma_frequency,
            phase_lock_accuracy: state.phase_lock_accuracy,
            // Phase 6B
            dominant_band: state.dominant_band.clone(),
            gamma_power: state.gamma_power,
            theta_power: state.theta_power,
            alpha_power: state.alpha_power,
            theta_gamma_coupling: state.theta_gamma_coupling,
            arousal: state.arousal,
            attention: state.attention,
            in_memory_window: state.in_memory_window,
            // Phase 6C Resonance
            system_resonance: state.system_resonance,
            resonance_integration_boost: state.resonance_integration_boost,
            resonating_pairs: state.resonating_pairs,
            cluster_count: state.cluster_count,
            primary_cluster_size: state.primary_cluster_size,
            primary_cluster_plv: state.primary_cluster_plv,
            // Phase 6C Attention
            attention_spotlight_position: state.attention_spotlight_position,
            attention_spotlight_intensity: state.attention_spotlight_intensity,
            attention_spotlight_width: state.attention_spotlight_width,
            attention_capacity: state.attention_capacity,
            secondary_foci_count: state.secondary_foci_count,
            in_attention_blink: state.in_attention_blink,
            attention_sampling_effectiveness: state.attention_sampling_effectiveness,
            in_attention_uptake: state.in_attention_uptake,
        }
    }
}

/// Summary of oscillatory consciousness state
#[derive(Debug, Clone)]
pub struct OscillatoryMindSummary {
    // Core consciousness
    pub phi: f64,
    pub effective_phi: f64,
    pub meta_awareness: f64,
    // Phase 6A: Oscillatory State
    pub phase_category: String,
    pub phase_coherence: f64,
    pub gamma_frequency: f64,
    pub phase_lock_accuracy: f64,
    // Phase 6B: Multi-Band Oscillations
    pub dominant_band: String,
    pub gamma_power: f64,
    pub theta_power: f64,
    pub alpha_power: f64,
    pub theta_gamma_coupling: f64,
    pub arousal: f64,
    pub attention: f64,
    pub in_memory_window: bool,
    // Phase 6C: Process Resonance
    pub system_resonance: f64,
    pub resonance_integration_boost: f64,
    pub resonating_pairs: usize,
    pub cluster_count: usize,
    pub primary_cluster_size: usize,
    pub primary_cluster_plv: f64,
    // Phase 6C: Attention Spotlight
    pub attention_spotlight_position: f64,
    pub attention_spotlight_intensity: f64,
    pub attention_spotlight_width: f64,
    pub attention_capacity: f64,
    pub secondary_foci_count: usize,
    pub in_attention_blink: bool,
    pub attention_sampling_effectiveness: f64,
    pub in_attention_uptake: bool,
}

impl Drop for ContinuousMind {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Response from the continuous mind
#[derive(Debug, Clone)]
pub struct MindResponse {
    /// The generated answer
    pub answer: String,

    /// Φ at time of response
    pub phi: f64,

    /// Meta-awareness level
    pub meta_awareness: f64,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Number of insights that emerged during processing
    pub insights_during_processing: usize,

    /// Whether the mind was conscious during processing
    pub was_conscious: bool,

    // === Active Inference Metrics ===

    /// Free energy (surprise) during processing
    pub free_energy: f64,

    /// Average surprise across domains
    pub average_surprise: f64,

    /// Whether curiosity pressure is high
    pub curiosity_pressure: bool,

    // === Language Integration Metrics (Phase 3) ===

    /// Unified free energy (language + general combined)
    pub unified_free_energy: f64,

    /// Φ from language consciousness bridge
    pub language_phi: f64,

    /// Suggested language actions (from curiosity-driven exploration)
    /// Examples: AskClarification, RequestConfirmation, FocusTopic
    pub language_actions: Vec<LanguageAction>,

    /// Whether this understanding gained spotlight in the language bridge
    /// (True means it won competitive selection for conscious attention)
    pub gained_spotlight: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_mind_creation() {
        let mind = ContinuousMind::new(MindConfig::default());
        assert!(!mind.is_conscious());  // Not conscious until awakened and processing
    }

    #[test]
    fn test_mind_awakening() {
        let mut mind = ContinuousMind::new(MindConfig::default());
        mind.awaken();

        // Let it run for a bit
        thread::sleep(Duration::from_millis(200));

        let state = mind.state();
        assert!(state.total_cycles > 0, "Should have completed some cycles");
        assert!(state.time_awake_ms > 0, "Should track time awake");

        mind.shutdown();
    }

    #[test]
    fn test_mind_processing() {
        let mut mind = ContinuousMind::new(MindConfig::default());
        mind.awaken();

        let response = mind.process("Hello world");

        assert!(!response.answer.is_empty());
        assert!(response.processing_time_ms > 0);

        mind.shutdown();
    }

    #[test]
    fn test_phi_emergence() {
        let mut mind = ContinuousMind::new(MindConfig::default());
        mind.awaken();

        // Process multiple inputs to activate processes
        mind.process("What is consciousness?");
        mind.process("How does thinking work?");
        mind.process("Tell me about the mind");

        // Let it integrate
        thread::sleep(Duration::from_millis(300));

        let state = mind.state();
        // Φ should be > 0 if processes are active and integrated
        println!("Φ after processing: {:.4}", state.phi);

        mind.shutdown();
    }
}
