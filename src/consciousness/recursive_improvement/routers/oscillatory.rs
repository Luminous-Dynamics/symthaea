//! # Oscillatory Phase-Locked Router
//!
//! Extends PredictiveRouter with phase awareness for true oscillatory routing.
//!
//! ## Key Concepts
//!
//! - **Multi-band oscillations**: Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-100Hz)
//! - **Cross-frequency coupling**: Theta-gamma nesting for hierarchical processing
//! - **Phase-locking**: Schedule operations at optimal phases
//! - **Coherence**: Track oscillatory stability across bands
//! - **Arousal modulation**: Arousal level shifts dominant frequency band
//!
//! ## Research Foundation
//!
//! - Gamma-band synchronization (Fries, 2005)
//! - Neural oscillations and consciousness (Buzsáki, 2006)
//! - Phase-amplitude coupling (Canolty & Knight, 2010)
//! - Theta-gamma coupling in memory (Lisman & Jensen, 2013)
//! - Cross-frequency interactions (Hyafil et al., 2015)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use super::predictive::{PredictiveRouter, PredictiveRouterConfig, PredictiveRouterSummary};

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-BAND OSCILLATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Frequency bands in neural oscillations
/// Each band serves distinct cognitive functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrequencyBand {
    // === PHASE 7: SLEEP-OSCILLATION BANDS ===
    /// Delta (0.5-4 Hz): Deep sleep, slow-wave activity, memory consolidation
    Delta,
    /// Sleep Spindle (12-14 Hz): Light sleep (N2), memory transfer to neocortex
    SleepSpindle,

    // === ORIGINAL WAKE-STATE BANDS ===
    /// Theta (4-8 Hz): Memory encoding, navigation, working memory, REM sleep
    Theta,
    /// Alpha (8-13 Hz): Idling, inhibition, attention gating
    Alpha,
    /// Beta (13-30 Hz): Motor planning, active maintenance, status quo
    Beta,
    /// Gamma (30-100 Hz): Binding, integration, conscious awareness
    Gamma,
}

impl FrequencyBand {
    /// Get the center frequency for this band
    pub fn center_frequency(&self) -> f64 {
        match self {
            // Phase 7: Sleep bands
            FrequencyBand::Delta => 2.0,         // 0.5-4 Hz center (deep sleep)
            FrequencyBand::SleepSpindle => 13.0, // 12-14 Hz (light sleep N2)
            // Wake-state bands
            FrequencyBand::Theta => 6.0,   // 4-8 Hz center
            FrequencyBand::Alpha => 10.0,  // 8-13 Hz center
            FrequencyBand::Beta => 20.0,   // 13-30 Hz center
            FrequencyBand::Gamma => 40.0,  // 30-100 Hz center
        }
    }

    /// Get the frequency range (min, max) for this band
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            // Phase 7: Sleep bands
            FrequencyBand::Delta => (0.5, 4.0),
            FrequencyBand::SleepSpindle => (12.0, 14.0),
            // Wake-state bands
            FrequencyBand::Theta => (4.0, 8.0),
            FrequencyBand::Alpha => (8.0, 13.0),
            FrequencyBand::Beta => (13.0, 30.0),
            FrequencyBand::Gamma => (30.0, 100.0),
        }
    }

    /// Get cognitive functions associated with this band
    pub fn cognitive_functions(&self) -> &'static [&'static str] {
        match self {
            // Phase 7: Sleep functions
            FrequencyBand::Delta => &["deep_sleep", "slow_wave_activity", "memory_consolidation", "synaptic_homeostasis"],
            FrequencyBand::SleepSpindle => &["light_sleep", "memory_transfer", "thalamo_cortical_gating", "sleep_protection"],
            // Wake-state functions
            FrequencyBand::Theta => &["memory_encoding", "navigation", "working_memory", "episodic_binding", "rem_dreaming"],
            FrequencyBand::Alpha => &["inhibition", "attention_gating", "idle_state", "sensory_suppression"],
            FrequencyBand::Beta => &["motor_planning", "status_quo", "active_maintenance", "top_down_control"],
            FrequencyBand::Gamma => &["binding", "integration", "consciousness", "attention_focus"],
        }
    }

    /// Get optimal arousal level for this band to dominate [0, 1]
    pub fn optimal_arousal(&self) -> f64 {
        match self {
            // Phase 7: Sleep states (very low arousal)
            FrequencyBand::Delta => 0.05,        // Near-zero arousal (deep sleep N3)
            FrequencyBand::SleepSpindle => 0.1,  // Very low arousal (light sleep N2)
            // Wake-state arousal levels
            FrequencyBand::Theta => 0.3,   // Low-medium arousal (drowsy, meditative, REM)
            FrequencyBand::Alpha => 0.2,   // Low arousal (relaxed, eyes closed)
            FrequencyBand::Beta => 0.6,    // Medium-high arousal (alert, focused)
            FrequencyBand::Gamma => 0.8,   // High arousal (intense focus, consciousness)
        }
    }

    /// Phase 7: Check if this band is a sleep-specific oscillation
    pub fn is_sleep_band(&self) -> bool {
        matches!(self, FrequencyBand::Delta | FrequencyBand::SleepSpindle)
    }

    /// Phase 7: Get the associated sleep stage (if any)
    pub fn sleep_stage(&self) -> Option<SleepStage> {
        match self {
            FrequencyBand::Delta => Some(SleepStage::N3),           // Deep sleep
            FrequencyBand::SleepSpindle => Some(SleepStage::N2),    // Light sleep
            FrequencyBand::Theta => Some(SleepStage::REM),          // REM (theta dominant)
            FrequencyBand::Alpha => Some(SleepStage::N1),           // Drowsy/light sleep onset
            FrequencyBand::Beta | FrequencyBand::Gamma => None,     // Wake-only bands
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 7: SLEEP-OSCILLATION INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Sleep stages following the AASM (American Academy of Sleep Medicine) classification
///
/// ## Research Foundation
/// - NREM stages N1-N3 (formerly stages 1-4)
/// - REM sleep with characteristic theta activity
/// - Sleep architecture and ultradian rhythms (~90 min cycles)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SleepStage {
    /// Awake: Alert consciousness with beta/gamma dominance
    Wake,
    /// N1 (Stage 1): Sleep onset, alpha dropout, theta emergence
    /// Duration: 5-10 minutes, easily awakened
    N1,
    /// N2 (Stage 2): Light sleep with sleep spindles and K-complexes
    /// Duration: 45-55% of total sleep, memory consolidation begins
    N2,
    /// N3 (Stage 3/4): Deep sleep, slow-wave activity (delta dominance)
    /// Duration: 15-25% of total sleep, synaptic homeostasis, immune function
    N3,
    /// REM: Rapid Eye Movement, dreaming, theta dominance, muscle atonia
    /// Duration: 20-25% of total sleep, emotional memory consolidation
    REM,
}

impl SleepStage {
    /// Get the dominant frequency band for this sleep stage
    pub fn dominant_band(&self) -> FrequencyBand {
        match self {
            SleepStage::Wake => FrequencyBand::Gamma,
            SleepStage::N1 => FrequencyBand::Alpha,      // Alpha dropout → theta
            SleepStage::N2 => FrequencyBand::SleepSpindle,
            SleepStage::N3 => FrequencyBand::Delta,
            SleepStage::REM => FrequencyBand::Theta,
        }
    }

    /// Get the target arousal level for this sleep stage
    pub fn target_arousal(&self) -> f64 {
        match self {
            SleepStage::Wake => 0.7,   // Alert
            SleepStage::N1 => 0.15,    // Drowsy
            SleepStage::N2 => 0.1,     // Light sleep
            SleepStage::N3 => 0.05,    // Deep sleep (minimal arousal)
            SleepStage::REM => 0.25,   // Paradoxical - brain active, body paralyzed
        }
    }

    /// Check if this is a sleep stage (not wake)
    pub fn is_sleeping(&self) -> bool {
        !matches!(self, SleepStage::Wake)
    }

    /// Check if this is NREM sleep
    pub fn is_nrem(&self) -> bool {
        matches!(self, SleepStage::N1 | SleepStage::N2 | SleepStage::N3)
    }

    /// Get memory consolidation strength for this stage [0, 1]
    pub fn consolidation_strength(&self) -> f64 {
        match self {
            SleepStage::Wake => 0.0,   // Encoding, not consolidating
            SleepStage::N1 => 0.1,     // Minimal consolidation
            SleepStage::N2 => 0.6,     // Spindle-mediated transfer
            SleepStage::N3 => 1.0,     // Maximum consolidation (slow-wave replay)
            SleepStage::REM => 0.7,    // Emotional and procedural memory
        }
    }

    /// Get consciousness level for this stage [0, 1]
    /// Based on IIT predictions: reduced Φ during deep sleep
    pub fn consciousness_level(&self) -> f64 {
        match self {
            SleepStage::Wake => 1.0,   // Full consciousness
            SleepStage::N1 => 0.5,     // Hypnagogic (partial awareness)
            SleepStage::N2 => 0.2,     // Reduced awareness
            SleepStage::N3 => 0.05,    // Minimal consciousness (near-zero Φ)
            SleepStage::REM => 0.7,    // Dream consciousness (vivid but disconnected)
        }
    }
}

/// Sleep-Oscillation Bridge connecting sleep state machine to oscillatory dynamics
///
/// ## Phase 7 Integration
///
/// This bridge synchronizes the SleepCycleManager (from brain/sleep.rs) with
/// the OscillatoryRouter's multi-band state. Sleep stages modulate:
///
/// 1. **Dominant frequency band** (Delta for N3, Theta for REM)
/// 2. **Cross-frequency coupling** (spindle-ripple in N2, theta-gamma in REM)
/// 3. **Arousal level** (minimal in N3, paradoxically high in REM)
/// 4. **Memory consolidation** (slow-wave replay, spindle-mediated transfer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepOscillationBridge {
    /// Current sleep stage
    pub current_stage: SleepStage,
    /// Previous sleep stage (for transition detection)
    pub previous_stage: SleepStage,
    /// Time in current stage (seconds)
    pub time_in_stage: f64,
    /// Current position in ultradian cycle [0, 1] (one ~90 min cycle)
    pub cycle_position: f64,
    /// Total sleep cycles completed
    pub cycles_completed: u32,
    /// Sleep pressure (homeostatic drive) [0, 1]
    pub sleep_pressure: f64,
    /// Circadian phase [0, 2π] (0 = midnight, π = noon)
    pub circadian_phase: f64,
    /// Memory consolidation queue size
    pub consolidation_queue_size: usize,
    /// Slow-wave ripple count (hippocampal sharp-wave ripples during N3)
    pub slow_wave_ripples: u64,
    /// Sleep spindle count during N2
    pub spindle_count: u64,
    /// REM density (eye movements per minute, proxy for dream intensity)
    pub rem_density: f64,
    /// Delta-spindle coupling strength (N2→N3 transition marker)
    pub delta_spindle_coupling: f64,
    /// Spindle-ripple coupling (memory transfer efficiency)
    pub spindle_ripple_coupling: f64,
}

impl SleepOscillationBridge {
    pub fn new() -> Self {
        Self {
            current_stage: SleepStage::Wake,
            previous_stage: SleepStage::Wake,
            time_in_stage: 0.0,
            cycle_position: 0.0,
            cycles_completed: 0,
            sleep_pressure: 0.0,
            circadian_phase: 0.0,
            consolidation_queue_size: 0,
            slow_wave_ripples: 0,
            spindle_count: 0,
            rem_density: 0.0,
            delta_spindle_coupling: 0.5,
            spindle_ripple_coupling: 0.5,
        }
    }

    /// Transition to a new sleep stage
    pub fn transition_to(&mut self, new_stage: SleepStage) {
        if new_stage != self.current_stage {
            self.previous_stage = self.current_stage;
            self.current_stage = new_stage;
            self.time_in_stage = 0.0;

            // Track sleep cycle completion (REM → N1/Wake = new cycle)
            if self.previous_stage == SleepStage::REM &&
               (new_stage == SleepStage::N1 || new_stage == SleepStage::Wake) {
                self.cycles_completed += 1;
            }
        }
    }

    /// Update the bridge state based on elapsed time
    pub fn update(&mut self, dt: f64, multi_band: &mut MultiBandState) {
        self.time_in_stage += dt;

        // Update sleep pressure (increases when awake, decreases when sleeping)
        if self.current_stage == SleepStage::Wake {
            self.sleep_pressure = (self.sleep_pressure + dt * 0.001).min(1.0);
        } else {
            let decay_rate = match self.current_stage {
                SleepStage::N3 => 0.005,  // Fast pressure relief in deep sleep
                SleepStage::N2 => 0.003,
                SleepStage::N1 => 0.001,
                SleepStage::REM => 0.002,
                SleepStage::Wake => 0.0,
            };
            self.sleep_pressure = (self.sleep_pressure - dt * decay_rate).max(0.0);
        }

        // Advance circadian phase (24-hour cycle)
        self.circadian_phase = (self.circadian_phase + dt * 2.0 * PI / 86400.0) % (2.0 * PI);

        // Update ultradian cycle position (90-minute cycles)
        if self.current_stage.is_sleeping() {
            self.cycle_position = (self.cycle_position + dt / 5400.0) % 1.0;
        }

        // === SLEEP STAGE → OSCILLATORY BAND MODULATION ===
        self.modulate_bands(multi_band);

        // === STAGE-SPECIFIC OSCILLATORY EVENTS ===
        match self.current_stage {
            SleepStage::N2 => {
                // Generate sleep spindles (12-14 Hz bursts, 0.5-2s duration)
                // Occur every 3-8 seconds during N2
                if self.time_in_stage % 5.0 < dt {
                    self.spindle_count += 1;
                    // Boost spindle band power briefly
                    if let Some(spindle_state) = multi_band.bands.get_mut(&FrequencyBand::SleepSpindle) {
                        spindle_state.power = (spindle_state.power + 0.3).min(1.0);
                    }
                }
            }
            SleepStage::N3 => {
                // Generate sharp-wave ripples during slow waves
                // These transfer memories from hippocampus to neocortex
                if self.time_in_stage % 3.0 < dt {
                    self.slow_wave_ripples += 1;
                }
                // Enhance delta-spindle coupling for memory consolidation
                self.delta_spindle_coupling = (self.delta_spindle_coupling + dt * 0.01).min(1.0);
            }
            SleepStage::REM => {
                // Update REM density (varies with dream intensity)
                // Higher in later REM periods of the night
                let base_density = 10.0 + (self.cycles_completed as f64) * 5.0;
                self.rem_density = base_density * (0.8 + 0.4 * rand_float(self.time_in_stage as u64));
            }
            _ => {}
        }
    }

    /// Modulate multi-band state based on current sleep stage
    fn modulate_bands(&self, multi_band: &mut MultiBandState) {
        let target_arousal = self.current_stage.target_arousal();

        // Smooth arousal transition
        multi_band.arousal = multi_band.arousal * 0.9 + target_arousal * 0.1;

        // Modulate band powers based on sleep stage
        match self.current_stage {
            SleepStage::Wake => {
                // Suppress sleep bands, enhance gamma
                self.suppress_band(multi_band, FrequencyBand::Delta, 0.1);
                self.suppress_band(multi_band, FrequencyBand::SleepSpindle, 0.1);
                self.enhance_band(multi_band, FrequencyBand::Gamma, 0.8);
                self.enhance_band(multi_band, FrequencyBand::Beta, 0.6);
            }
            SleepStage::N1 => {
                // Alpha dropout, theta emergence
                self.suppress_band(multi_band, FrequencyBand::Alpha, 0.3);
                self.enhance_band(multi_band, FrequencyBand::Theta, 0.5);
                self.suppress_band(multi_band, FrequencyBand::Gamma, 0.2);
            }
            SleepStage::N2 => {
                // Sleep spindles prominent, some delta
                self.enhance_band(multi_band, FrequencyBand::SleepSpindle, 0.7);
                self.enhance_band(multi_band, FrequencyBand::Delta, 0.3);
                self.suppress_band(multi_band, FrequencyBand::Gamma, 0.1);
            }
            SleepStage::N3 => {
                // Delta dominance (slow-wave sleep)
                self.enhance_band(multi_band, FrequencyBand::Delta, 0.9);
                self.suppress_band(multi_band, FrequencyBand::SleepSpindle, 0.2);
                self.suppress_band(multi_band, FrequencyBand::Theta, 0.1);
                self.suppress_band(multi_band, FrequencyBand::Gamma, 0.05);
            }
            SleepStage::REM => {
                // Theta dominance (like waking theta), suppress delta
                self.enhance_band(multi_band, FrequencyBand::Theta, 0.8);
                self.enhance_band(multi_band, FrequencyBand::Gamma, 0.4); // Dream binding
                self.suppress_band(multi_band, FrequencyBand::Delta, 0.1);
                // Note: Beta/motor suppressed (REM atonia)
                self.suppress_band(multi_band, FrequencyBand::Beta, 0.1);
            }
        }

        // Update dominant band based on sleep stage
        multi_band.dominant_band = self.current_stage.dominant_band();
    }

    /// Helper: Enhance a band's power toward target
    fn enhance_band(&self, multi_band: &mut MultiBandState, band: FrequencyBand, target: f64) {
        if let Some(state) = multi_band.bands.get_mut(&band) {
            state.power = state.power * 0.8 + target * 0.2;
        }
    }

    /// Helper: Suppress a band's power toward target
    fn suppress_band(&self, multi_band: &mut MultiBandState, band: FrequencyBand, target: f64) {
        if let Some(state) = multi_band.bands.get_mut(&band) {
            state.power = state.power * 0.8 + target * 0.2;
        }
    }

    /// Get memory consolidation efficiency for current state
    pub fn consolidation_efficiency(&self) -> f64 {
        let base = self.current_stage.consolidation_strength();

        // Boost efficiency based on spindle-ripple coupling
        let coupling_boost = if self.current_stage == SleepStage::N2 || self.current_stage == SleepStage::N3 {
            self.spindle_ripple_coupling * 0.2
        } else {
            0.0
        };

        (base + coupling_boost).min(1.0)
    }

    /// Check if currently in a memory transfer window (optimal for consolidation)
    pub fn in_memory_transfer_window(&self) -> bool {
        match self.current_stage {
            SleepStage::N2 => self.spindle_count % 3 == 0, // Every third spindle
            SleepStage::N3 => self.slow_wave_ripples % 2 == 0, // Every other ripple
            _ => false,
        }
    }

    /// Get dream intensity (REM only) [0, 1]
    pub fn dream_intensity(&self) -> f64 {
        if self.current_stage == SleepStage::REM {
            (self.rem_density / 30.0).min(1.0) // Normalize to ~30 movements/min max
        } else {
            0.0
        }
    }

    /// Get summary for MindState integration
    pub fn summary(&self) -> SleepOscillationSummary {
        SleepOscillationSummary {
            stage: self.current_stage,
            stage_name: format!("{:?}", self.current_stage),
            time_in_stage_secs: self.time_in_stage,
            sleep_pressure: self.sleep_pressure,
            cycles_completed: self.cycles_completed,
            cycle_position: self.cycle_position,
            consolidation_efficiency: self.consolidation_efficiency(),
            in_memory_window: self.in_memory_transfer_window(),
            consciousness_level: self.current_stage.consciousness_level(),
            spindle_count: self.spindle_count,
            ripple_count: self.slow_wave_ripples,
            dream_intensity: self.dream_intensity(),
            is_sleeping: self.current_stage.is_sleeping(),
        }
    }
}

impl Default for SleepOscillationBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of sleep-oscillation state for MindState dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepOscillationSummary {
    /// Current sleep stage
    pub stage: SleepStage,
    /// Human-readable stage name
    pub stage_name: String,
    /// Time in current stage (seconds)
    pub time_in_stage_secs: f64,
    /// Sleep pressure (homeostatic drive) [0, 1]
    pub sleep_pressure: f64,
    /// Completed ultradian cycles
    pub cycles_completed: u32,
    /// Position in current cycle [0, 1]
    pub cycle_position: f64,
    /// Memory consolidation efficiency [0, 1]
    pub consolidation_efficiency: f64,
    /// Currently in optimal memory transfer window
    pub in_memory_window: bool,
    /// Consciousness level [0, 1]
    pub consciousness_level: f64,
    /// Sleep spindle count (N2)
    pub spindle_count: u64,
    /// Sharp-wave ripple count (N3)
    pub ripple_count: u64,
    /// Dream intensity (REM) [0, 1]
    pub dream_intensity: f64,
    /// Is currently sleeping
    pub is_sleeping: bool,
}

/// Simple random float for REM density variation
fn rand_float(seed: u64) -> f64 {
    let x = seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
    ((x >> 17) as f64) / (u32::MAX as f64)
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 8: NEUROMODULATORY DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════
//
// Neuromodulators are the "global broadcast" chemicals that modulate brain-wide
// activity patterns, consciousness states, and cognitive functions.
//
// ## Research Foundation
// - Dopamine: Reward, motivation, learning (Schultz, 1997; Berridge, 2007)
// - Serotonin: Mood, impulse control, well-being (Cools et al., 2008)
// - Norepinephrine: Arousal, alertness, attention (Aston-Jones & Cohen, 2005)
// - Acetylcholine: Learning, attention, memory (Hasselmo, 2006)
// - Neuromodulatory orchestration (Marder, 2012; Dayan, 2012)
// ═══════════════════════════════════════════════════════════════════════════

/// The four major neuromodulatory systems affecting consciousness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Neuromodulator {
    /// Dopamine (DA): Reward prediction, motivation, motor control, learning
    /// - High: Motivation, pleasure, focused pursuit of goals
    /// - Low: Anhedonia, lack of motivation, difficulty with reward learning
    Dopamine,

    /// Serotonin (5-HT): Mood regulation, impulse control, satiety, well-being
    /// - High: Contentment, patience, impulse control
    /// - Low: Depression, impulsivity, anxiety
    Serotonin,

    /// Norepinephrine (NE): Arousal, alertness, stress response, attention
    /// - High: Alert, focused, stress response active
    /// - Low: Drowsy, inattentive, relaxed
    Norepinephrine,

    /// Acetylcholine (ACh): Learning, memory encoding, attention, REM sleep
    /// - High: Enhanced learning, vivid dreams (REM), focused attention
    /// - Low: Memory difficulties, reduced plasticity
    Acetylcholine,
}

impl Neuromodulator {
    /// Get all neuromodulators
    pub fn all() -> [Neuromodulator; 4] {
        [
            Neuromodulator::Dopamine,
            Neuromodulator::Serotonin,
            Neuromodulator::Norepinephrine,
            Neuromodulator::Acetylcholine,
        ]
    }

    /// Get the short code for this neuromodulator
    pub fn code(&self) -> &'static str {
        match self {
            Neuromodulator::Dopamine => "DA",
            Neuromodulator::Serotonin => "5-HT",
            Neuromodulator::Norepinephrine => "NE",
            Neuromodulator::Acetylcholine => "ACh",
        }
    }

    /// Get baseline concentration (tonic level) [0, 1]
    pub fn baseline(&self) -> f64 {
        match self {
            Neuromodulator::Dopamine => 0.3,      // Moderate baseline, phasic bursts for rewards
            Neuromodulator::Serotonin => 0.5,     // Higher tonic level for mood stability
            Neuromodulator::Norepinephrine => 0.2, // Low at rest, high during arousal
            Neuromodulator::Acetylcholine => 0.4,  // Moderate, varies with sleep/wake
        }
    }

    /// Get reuptake rate (how fast concentration returns to baseline) [0, 1]
    /// Higher = faster return to baseline
    pub fn reuptake_rate(&self) -> f64 {
        match self {
            Neuromodulator::Dopamine => 0.15,      // Fast reuptake (DAT transporter)
            Neuromodulator::Serotonin => 0.08,     // Slower (SERT transporter)
            Neuromodulator::Norepinephrine => 0.12, // Medium (NET transporter)
            Neuromodulator::Acetylcholine => 0.25,  // Fast (acetylcholinesterase)
        }
    }

    /// Get release rate multiplier for stimulus intensity
    pub fn release_gain(&self) -> f64 {
        match self {
            Neuromodulator::Dopamine => 2.0,       // Strong phasic response to rewards
            Neuromodulator::Serotonin => 0.8,      // More stable, slower changes
            Neuromodulator::Norepinephrine => 1.5, // Quick response to arousal
            Neuromodulator::Acetylcholine => 1.2,  // Moderate response
        }
    }

    /// Get primary cognitive effects
    pub fn effects(&self) -> &'static [&'static str] {
        match self {
            Neuromodulator::Dopamine => &["motivation", "reward_learning", "motor_control", "goal_pursuit", "pleasure"],
            Neuromodulator::Serotonin => &["mood", "impulse_control", "satiety", "patience", "well_being"],
            Neuromodulator::Norepinephrine => &["arousal", "alertness", "attention", "stress_response", "vigilance"],
            Neuromodulator::Acetylcholine => &["learning", "memory_encoding", "attention", "rem_sleep", "plasticity"],
        }
    }

    /// Get effect on oscillatory bands (which bands are enhanced/suppressed)
    /// Returns (enhanced_bands, suppressed_bands)
    pub fn oscillatory_effects(&self) -> (&'static [FrequencyBand], &'static [FrequencyBand]) {
        match self {
            Neuromodulator::Dopamine => (
                &[FrequencyBand::Beta, FrequencyBand::Gamma],  // Enhances high-frequency for motivation
                &[FrequencyBand::Alpha]                         // Suppresses idle state
            ),
            Neuromodulator::Serotonin => (
                &[FrequencyBand::Alpha],                        // Promotes calm, stable state
                &[FrequencyBand::Beta, FrequencyBand::Gamma]   // Reduces agitation
            ),
            Neuromodulator::Norepinephrine => (
                &[FrequencyBand::Beta, FrequencyBand::Gamma],  // Enhances alertness
                &[FrequencyBand::Delta, FrequencyBand::Alpha]  // Suppresses drowsiness
            ),
            Neuromodulator::Acetylcholine => (
                &[FrequencyBand::Theta, FrequencyBand::Gamma], // Learning and memory
                &[FrequencyBand::Delta]                         // Opposes deep sleep (during wake)
            ),
        }
    }

    /// Get sleep-wake modulation pattern
    /// Returns (wake_level, nrem_level, rem_level)
    pub fn sleep_pattern(&self) -> (f64, f64, f64) {
        match self {
            Neuromodulator::Dopamine => (0.5, 0.2, 0.3),       // Active during wake, low in sleep
            Neuromodulator::Serotonin => (0.6, 0.3, 0.05),     // High wake, drops in REM (REM-off)
            Neuromodulator::Norepinephrine => (0.6, 0.1, 0.0), // Wake-promoting, silent in REM
            Neuromodulator::Acetylcholine => (0.5, 0.1, 0.8),  // Low in NREM, HIGH in REM!
        }
    }
}

/// State of a single neuromodulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulatorState {
    /// The neuromodulator type
    pub modulator: Neuromodulator,
    /// Current concentration [0, 1] (0 = depleted, 1 = maximum)
    pub concentration: f64,
    /// Rate of change (derivative)
    pub rate_of_change: f64,
    /// Receptor sensitivity (adapts over time) [0, 1]
    pub receptor_sensitivity: f64,
    /// Cumulative release (for depletion tracking)
    pub cumulative_release: f64,
    /// Time since last major release event
    pub time_since_release: f64,
}

impl NeuromodulatorState {
    pub fn new(modulator: Neuromodulator) -> Self {
        Self {
            modulator,
            concentration: modulator.baseline(),
            rate_of_change: 0.0,
            receptor_sensitivity: 1.0,
            cumulative_release: 0.0,
            time_since_release: 0.0,
        }
    }

    /// Effective level = concentration × receptor_sensitivity
    pub fn effective_level(&self) -> f64 {
        (self.concentration * self.receptor_sensitivity).clamp(0.0, 1.0)
    }

    /// Release neuromodulator (phasic burst)
    pub fn release(&mut self, intensity: f64) {
        let release_amount = intensity * self.modulator.release_gain() * 0.1;
        self.concentration = (self.concentration + release_amount).min(1.0);
        self.cumulative_release += release_amount;
        self.time_since_release = 0.0;
        self.rate_of_change = release_amount;

        // Receptor down-regulation from high release
        if self.concentration > 0.8 {
            self.receptor_sensitivity *= 0.995; // Gradual desensitization
        }
    }

    /// Update with reuptake dynamics (call each timestep)
    pub fn update(&mut self, dt: f64) {
        let baseline = self.modulator.baseline();
        let reuptake = self.modulator.reuptake_rate();

        // Exponential decay toward baseline
        let delta = (baseline - self.concentration) * reuptake * dt;
        self.concentration = (self.concentration + delta).clamp(0.0, 1.0);
        self.rate_of_change = delta / dt;
        self.time_since_release += dt;

        // Receptor re-sensitization when concentration is normal
        if self.concentration < 0.6 && self.receptor_sensitivity < 1.0 {
            self.receptor_sensitivity = (self.receptor_sensitivity + 0.001 * dt).min(1.0);
        }
    }

    /// Set concentration directly (for sleep stage transitions)
    pub fn set_concentration(&mut self, level: f64) {
        let old = self.concentration;
        self.concentration = level.clamp(0.0, 1.0);
        self.rate_of_change = self.concentration - old;
    }
}

/// Complete neuromodulatory system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulatorSystem {
    /// Individual neuromodulator states
    pub dopamine: NeuromodulatorState,
    pub serotonin: NeuromodulatorState,
    pub norepinephrine: NeuromodulatorState,
    pub acetylcholine: NeuromodulatorState,

    /// Emergent states from neuromodulator balance
    pub mood_valence: f64,          // -1 (negative) to +1 (positive)
    pub arousal_level: f64,         // 0 (drowsy) to 1 (hyperaroused)
    pub motivation_drive: f64,      // 0 (amotivated) to 1 (highly driven)
    pub learning_rate_mod: f64,     // Multiplier on learning rate [0.5, 2.0]
    pub stress_level: f64,          // 0 (calm) to 1 (stressed)

    /// Interaction tracking
    time_since_update: f64,
}

impl NeuromodulatorSystem {
    pub fn new() -> Self {
        Self {
            dopamine: NeuromodulatorState::new(Neuromodulator::Dopamine),
            serotonin: NeuromodulatorState::new(Neuromodulator::Serotonin),
            norepinephrine: NeuromodulatorState::new(Neuromodulator::Norepinephrine),
            acetylcholine: NeuromodulatorState::new(Neuromodulator::Acetylcholine),
            mood_valence: 0.2,        // Slightly positive baseline
            arousal_level: 0.4,       // Moderate arousal
            motivation_drive: 0.5,    // Neutral motivation
            learning_rate_mod: 1.0,   // Normal learning
            stress_level: 0.2,        // Low stress
            time_since_update: 0.0,
        }
    }

    /// Get neuromodulator by type
    pub fn get(&self, nm: Neuromodulator) -> &NeuromodulatorState {
        match nm {
            Neuromodulator::Dopamine => &self.dopamine,
            Neuromodulator::Serotonin => &self.serotonin,
            Neuromodulator::Norepinephrine => &self.norepinephrine,
            Neuromodulator::Acetylcholine => &self.acetylcholine,
        }
    }

    /// Get mutable neuromodulator by type
    pub fn get_mut(&mut self, nm: Neuromodulator) -> &mut NeuromodulatorState {
        match nm {
            Neuromodulator::Dopamine => &mut self.dopamine,
            Neuromodulator::Serotonin => &mut self.serotonin,
            Neuromodulator::Norepinephrine => &mut self.norepinephrine,
            Neuromodulator::Acetylcholine => &mut self.acetylcholine,
        }
    }

    /// Release a specific neuromodulator
    pub fn release(&mut self, nm: Neuromodulator, intensity: f64) {
        self.get_mut(nm).release(intensity);
    }

    /// Process a reward signal (affects dopamine primarily)
    pub fn process_reward(&mut self, reward: f64, prediction_error: f64) {
        // Dopamine encodes reward prediction error (Schultz, 1997)
        let da_release = (prediction_error * 2.0).clamp(-0.5, 1.0);
        if da_release > 0.0 {
            self.dopamine.release(da_release);
        } else {
            // Negative prediction error causes dopamine dip
            self.dopamine.concentration = (self.dopamine.concentration + da_release * 0.1).max(0.0);
        }

        // Positive rewards also boost serotonin slightly
        if reward > 0.5 {
            self.serotonin.release(reward * 0.3);
        }
    }

    /// Process a stress/threat signal
    pub fn process_stress(&mut self, stress_intensity: f64) {
        // Norepinephrine surges during stress
        self.norepinephrine.release(stress_intensity);

        // High stress depletes serotonin over time
        if stress_intensity > 0.5 {
            self.serotonin.concentration *= 0.98;
        }

        // Acetylcholine increases attention during stress
        self.acetylcholine.release(stress_intensity * 0.5);
    }

    /// Process novelty/curiosity signal
    pub fn process_novelty(&mut self, novelty: f64) {
        // Novelty triggers dopamine (wanting) and norepinephrine (alertness)
        self.dopamine.release(novelty * 0.6);
        self.norepinephrine.release(novelty * 0.4);
        self.acetylcholine.release(novelty * 0.3); // Enhance learning
    }

    /// Adapt to sleep stage
    pub fn adapt_to_sleep_stage(&mut self, stage: SleepStage) {
        let (wake, nrem, rem) = match stage {
            SleepStage::Wake => (1.0, 0.0, 0.0),
            SleepStage::N1 => (0.5, 0.5, 0.0),
            SleepStage::N2 => (0.2, 0.8, 0.0),
            SleepStage::N3 => (0.0, 1.0, 0.0),
            SleepStage::REM => (0.0, 0.0, 1.0),
        };

        for nm in Neuromodulator::all() {
            let (wake_level, nrem_level, rem_level) = nm.sleep_pattern();
            let target = wake * wake_level + nrem * nrem_level + rem * rem_level;
            let current = self.get(nm).concentration;
            let new_level = current * 0.9 + target * 0.1; // Smooth transition
            self.get_mut(nm).set_concentration(new_level);
        }
    }

    /// Update all dynamics (call each timestep)
    pub fn update(&mut self, dt: f64) {
        // Update individual neuromodulators
        self.dopamine.update(dt);
        self.serotonin.update(dt);
        self.norepinephrine.update(dt);
        self.acetylcholine.update(dt);

        // Compute emergent states
        self.compute_emergent_states();

        self.time_since_update = 0.0;
    }

    /// Compute emergent psychological states from neuromodulator balance
    fn compute_emergent_states(&mut self) {
        let da = self.dopamine.effective_level();
        let ht = self.serotonin.effective_level();
        let ne = self.norepinephrine.effective_level();
        let ach = self.acetylcholine.effective_level();

        // Mood valence: Serotonin + Dopamine (positive), inverse of stress
        // Range: -1 to +1
        self.mood_valence = ((da * 0.4 + ht * 0.6) - 0.3).clamp(-1.0, 1.0) * 2.0;

        // Arousal level: Norepinephrine + Acetylcholine
        self.arousal_level = (ne * 0.6 + ach * 0.3 + da * 0.1).clamp(0.0, 1.0);

        // Motivation drive: Dopamine (wanting) modulated by serotonin (patience)
        self.motivation_drive = (da * 0.7 + ht * 0.2 + ne * 0.1).clamp(0.0, 1.0);

        // Learning rate modifier: Acetylcholine + Dopamine (reward learning)
        // Range: 0.5 to 2.0
        self.learning_rate_mod = 0.5 + (ach * 0.8 + da * 0.5).clamp(0.0, 1.5);

        // Stress level: High NE + Low serotonin = stress
        self.stress_level = (ne * 0.7 - ht * 0.3 + 0.15).clamp(0.0, 1.0);
    }

    /// Get modulation effect on a specific frequency band
    pub fn band_modulation(&self, band: FrequencyBand) -> f64 {
        let mut modulation = 1.0;

        for nm in Neuromodulator::all() {
            let level = self.get(nm).effective_level();
            let (enhanced, suppressed) = nm.oscillatory_effects();

            if enhanced.contains(&band) {
                modulation *= 1.0 + level * 0.3; // Up to 30% enhancement
            }
            if suppressed.contains(&band) {
                modulation *= 1.0 - level * 0.2; // Up to 20% suppression
            }
        }

        modulation.clamp(0.5, 2.0)
    }

    /// Get summary for dashboard
    pub fn summary(&self) -> NeuromodulatorSummary {
        NeuromodulatorSummary {
            dopamine_level: self.dopamine.effective_level(),
            serotonin_level: self.serotonin.effective_level(),
            norepinephrine_level: self.norepinephrine.effective_level(),
            acetylcholine_level: self.acetylcholine.effective_level(),
            dopamine_sensitivity: self.dopamine.receptor_sensitivity,
            serotonin_sensitivity: self.serotonin.receptor_sensitivity,
            mood_valence: self.mood_valence,
            arousal_level: self.arousal_level,
            motivation_drive: self.motivation_drive,
            learning_rate_mod: self.learning_rate_mod,
            stress_level: self.stress_level,
        }
    }
}

impl Default for NeuromodulatorSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of neuromodulatory state for MindState dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodulatorSummary {
    // Concentration levels [0, 1]
    pub dopamine_level: f64,
    pub serotonin_level: f64,
    pub norepinephrine_level: f64,
    pub acetylcholine_level: f64,

    // Receptor sensitivities [0, 1]
    pub dopamine_sensitivity: f64,
    pub serotonin_sensitivity: f64,

    // Emergent states
    pub mood_valence: f64,       // -1 to +1
    pub arousal_level: f64,      // 0 to 1
    pub motivation_drive: f64,   // 0 to 1
    pub learning_rate_mod: f64,  // 0.5 to 2.0
    pub stress_level: f64,       // 0 to 1
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 9: PREDICTIVE CONSCIOUSNESS INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════
//
// Implements Friston's Free Energy Principle within the oscillatory framework:
// - Free energy tracking tied to oscillatory state
// - Precision weighting influenced by neuromodulators
// - Prediction error propagation across frequency bands
// - Active inference-driven arousal modulation
//
// Research Foundation:
// - Free Energy Principle (Friston, 2010)
// - Predictive Processing (Clark, 2013)
// - Neural Precision (Feldman & Friston, 2010)
// - Dopamine and Precision (Fries, 2015)

/// Hierarchical level for predictive processing within consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveLevel {
    /// Level name (sensory, perceptual, conceptual, metacognitive)
    pub name: &'static str,
    /// Current prediction at this level
    pub prediction: f64,
    /// Prediction error (mismatch between prediction and input)
    pub prediction_error: f64,
    /// Precision (confidence in prediction) - influenced by neuromodulators
    pub precision: f64,
    /// Free energy contribution from this level
    pub free_energy: f64,
    /// Learning rate for this level
    pub learning_rate: f64,
}

impl PredictiveLevel {
    pub fn new(name: &'static str, level_idx: usize) -> Self {
        // Higher levels have lower learning rates and start with higher precision
        let learning_rate = 0.1 / (1.0 + 0.3 * level_idx as f64);
        let precision = 0.5 + 0.1 * level_idx as f64;

        Self {
            name,
            prediction: 0.5,
            prediction_error: 0.0,
            precision: precision.min(1.0),
            free_energy: 0.0,
            learning_rate,
        }
    }

    /// Update prediction based on error (gradient descent on free energy)
    pub fn update(&mut self, input: f64, dt: f64) {
        // Compute prediction error
        self.prediction_error = input - self.prediction;

        // Free energy = precision-weighted squared error (variational free energy)
        self.free_energy = 0.5 * self.precision * self.prediction_error.powi(2);

        // Update prediction to minimize free energy
        let delta = self.learning_rate * self.precision * self.prediction_error * dt;
        self.prediction = (self.prediction + delta).clamp(0.0, 1.0);
    }

    /// Update precision based on recent error history (precision as inverse variance)
    pub fn adapt_precision(&mut self, error_variance: f64) {
        // Precision is inverse of expected error variance
        // Higher variance → lower precision (less confident)
        let new_precision = 1.0 / (1.0 + error_variance);
        // Slow adaptation
        self.precision = 0.9 * self.precision + 0.1 * new_precision;
    }

    /// Apply neuromodulator effects on precision
    /// Dopamine: increases precision on salient/rewarding stimuli
    /// Norepinephrine: globally increases precision (arousal)
    /// Acetylcholine: increases precision for learning
    pub fn modulate_precision(&mut self, da: f64, ne: f64, ach: f64) {
        // Baseline precision modulation
        let modulation = 1.0 + 0.3 * (da - 0.3) + 0.2 * (ne - 0.2) + 0.2 * (ach - 0.4);
        self.precision = (self.precision * modulation).clamp(0.1, 2.0);
    }
}

/// Complete state of predictive consciousness system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveConsciousnessState {
    // === Hierarchical Levels ===
    /// Sensory level (raw input processing)
    pub sensory: PredictiveLevel,
    /// Perceptual level (feature integration)
    pub perceptual: PredictiveLevel,
    /// Conceptual level (abstract representations)
    pub conceptual: PredictiveLevel,
    /// Metacognitive level (awareness of awareness)
    pub metacognitive: PredictiveLevel,

    // === Global Free Energy ===
    /// Total free energy across all levels
    pub total_free_energy: f64,
    /// Complexity term (divergence from prior)
    pub complexity: f64,
    /// Accuracy term (prediction error)
    pub accuracy: f64,

    // === Precision Weighting ===
    /// Global precision gain (influenced by arousal)
    pub precision_gain: f64,
    /// Precision-weighted prediction error
    pub weighted_prediction_error: f64,

    // === Active Inference ===
    /// Expected free energy for current action
    pub expected_free_energy: f64,
    /// Epistemic value (information gain potential)
    pub epistemic_value: f64,
    /// Pragmatic value (goal-alignment)
    pub pragmatic_value: f64,
    /// Exploration-exploitation balance [0=exploit, 1=explore]
    pub exploration_drive: f64,

    // === Error History ===
    /// Running average of prediction error
    pub mean_error: f64,
    /// Running variance of prediction error
    pub error_variance: f64,
    /// Surprise (negative log probability of observation)
    pub surprise: f64,

    // === Time ===
    /// Total updates
    pub updates: u64,
}

impl PredictiveConsciousnessState {
    pub fn new() -> Self {
        Self {
            sensory: PredictiveLevel::new("sensory", 0),
            perceptual: PredictiveLevel::new("perceptual", 1),
            conceptual: PredictiveLevel::new("conceptual", 2),
            metacognitive: PredictiveLevel::new("metacognitive", 3),

            total_free_energy: 0.0,
            complexity: 0.0,
            accuracy: 0.0,

            precision_gain: 1.0,
            weighted_prediction_error: 0.0,

            expected_free_energy: 0.0,
            epistemic_value: 0.0,
            pragmatic_value: 0.0,
            exploration_drive: 0.5,

            mean_error: 0.0,
            error_variance: 0.01,
            surprise: 0.0,

            updates: 0,
        }
    }

    /// Process observation through the hierarchy
    /// phi: integrated information (bottom-up signal)
    /// coherence: oscillatory coherence
    /// meta: meta-awareness level
    pub fn observe(&mut self, phi: f64, coherence: f64, meta: f64, dt: f64) {
        // Bottom-up pass: sensory → perceptual → conceptual → metacognitive
        self.sensory.update(phi, dt);
        self.perceptual.update(self.sensory.prediction, dt);
        self.conceptual.update(self.perceptual.prediction + coherence * 0.3, dt);
        self.metacognitive.update(self.conceptual.prediction + meta * 0.2, dt);

        // Compute total prediction error (precision-weighted)
        let total_error =
            self.sensory.precision * self.sensory.prediction_error.abs() +
            self.perceptual.precision * self.perceptual.prediction_error.abs() +
            self.conceptual.precision * self.conceptual.prediction_error.abs() +
            self.metacognitive.precision * self.metacognitive.prediction_error.abs();

        self.weighted_prediction_error = total_error / 4.0;

        // Compute free energy
        self.accuracy = total_error;
        self.complexity = self.compute_complexity();
        self.total_free_energy = self.accuracy + self.complexity;

        // Update surprise (negative log probability)
        self.surprise = (1.0 + self.weighted_prediction_error).ln();

        // Update running error statistics
        let alpha = 0.95;
        self.mean_error = alpha * self.mean_error + (1.0 - alpha) * self.weighted_prediction_error;
        let squared_diff = (self.weighted_prediction_error - self.mean_error).powi(2);
        self.error_variance = alpha * self.error_variance + (1.0 - alpha) * squared_diff;

        // Adapt precisions based on error variance
        self.sensory.adapt_precision(self.error_variance);
        self.perceptual.adapt_precision(self.error_variance * 1.2);
        self.conceptual.adapt_precision(self.error_variance * 1.4);
        self.metacognitive.adapt_precision(self.error_variance * 1.6);

        self.updates += 1;
    }

    /// Apply neuromodulator effects to precision weighting
    pub fn apply_neuromodulators(&mut self, da: f64, _serotonin: f64, ne: f64, ach: f64) {
        // Global precision gain from norepinephrine (arousal → precision)
        self.precision_gain = 1.0 + 0.5 * (ne - 0.2).max(0.0);

        // Modulate each level
        self.sensory.modulate_precision(da * 0.5, ne, ach);
        self.perceptual.modulate_precision(da * 0.7, ne, ach);
        self.conceptual.modulate_precision(da, ne, ach * 0.8);
        self.metacognitive.modulate_precision(da * 1.2, ne * 0.8, ach * 0.6);
    }

    /// Compute expected free energy for potential action
    pub fn compute_expected_free_energy(&mut self, goal_alignment: f64, uncertainty: f64) {
        // Pragmatic value: how well does predicted outcome match preferences
        self.pragmatic_value = goal_alignment;

        // Epistemic value: expected information gain (reduction in uncertainty)
        self.epistemic_value = uncertainty * 0.5;

        // Expected free energy = -pragmatic + epistemic (want low EFE)
        self.expected_free_energy = -self.pragmatic_value + self.epistemic_value;

        // Exploration drive based on uncertainty vs goal alignment
        if goal_alignment > 0.7 {
            // Good alignment → exploit
            self.exploration_drive = 0.2;
        } else if uncertainty > 0.5 {
            // High uncertainty → explore
            self.exploration_drive = 0.8;
        } else {
            // Balance
            self.exploration_drive = 0.5 - goal_alignment * 0.3 + uncertainty * 0.3;
        }
        self.exploration_drive = self.exploration_drive.clamp(0.0, 1.0);
    }

    /// Compute complexity term (KL divergence from prior)
    fn compute_complexity(&self) -> f64 {
        // Prior: predictions near 0.5 with high variance
        let prior_mean = 0.5;

        // Complexity = KL(q || p) ≈ sum of squared deviations from prior
        let sensory_dev = (self.sensory.prediction - prior_mean).powi(2);
        let perceptual_dev = (self.perceptual.prediction - prior_mean).powi(2);
        let conceptual_dev = (self.conceptual.prediction - prior_mean).powi(2);
        let meta_dev = (self.metacognitive.prediction - prior_mean).powi(2);

        0.1 * (sensory_dev + perceptual_dev + conceptual_dev + meta_dev)
    }

    /// Get summary for dashboard/MindState
    pub fn summary(&self) -> PredictiveConsciousnessSummary {
        PredictiveConsciousnessSummary {
            total_free_energy: self.total_free_energy,
            complexity: self.complexity,
            accuracy: self.accuracy,
            precision_gain: self.precision_gain,
            weighted_prediction_error: self.weighted_prediction_error,
            expected_free_energy: self.expected_free_energy,
            epistemic_value: self.epistemic_value,
            pragmatic_value: self.pragmatic_value,
            exploration_drive: self.exploration_drive,
            surprise: self.surprise,
            sensory_precision: self.sensory.precision,
            perceptual_precision: self.perceptual.precision,
            conceptual_precision: self.conceptual.precision,
            metacognitive_precision: self.metacognitive.precision,
            sensory_prediction: self.sensory.prediction,
            perceptual_prediction: self.perceptual.prediction,
            conceptual_prediction: self.conceptual.prediction,
            metacognitive_prediction: self.metacognitive.prediction,
            updates: self.updates,
        }
    }
}

impl Default for PredictiveConsciousnessState {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of predictive consciousness state for MindState integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveConsciousnessSummary {
    // Free energy components
    pub total_free_energy: f64,
    pub complexity: f64,
    pub accuracy: f64,

    // Precision
    pub precision_gain: f64,
    pub weighted_prediction_error: f64,

    // Active inference
    pub expected_free_energy: f64,
    pub epistemic_value: f64,
    pub pragmatic_value: f64,
    pub exploration_drive: f64,

    // Current state
    pub surprise: f64,

    // Level precisions
    pub sensory_precision: f64,
    pub perceptual_precision: f64,
    pub conceptual_precision: f64,
    pub metacognitive_precision: f64,

    // Level predictions
    pub sensory_prediction: f64,
    pub perceptual_prediction: f64,
    pub conceptual_prediction: f64,
    pub metacognitive_prediction: f64,

    // Statistics
    pub updates: u64,
}

// =============================================================================
// GLOBAL WORKSPACE STATE (Phase 10)
// =============================================================================
//
// Bernard Baars' Global Workspace Theory (GWT) models consciousness as:
// 1. COMPETITION: Multiple specialized processors compete for workspace access
// 2. IGNITION: When activation crosses threshold, content "ignites" into consciousness
// 3. BROADCAST: Winning content is globally broadcast to ALL processors
// 4. FADING: Without sustained activation, content fades from consciousness
//
// Integration with oscillatory dynamics:
// - Gamma synchronization IS the broadcast mechanism (neurophysiology)
// - Attention spotlight determines what competes (Phase 6C)
// - Neuromodulators affect ignition threshold (Phase 8)
// - Predictive errors can trigger ignition (Phase 9)

/// A specialized processor module competing for global workspace access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkspaceProcessor {
    /// Perceptual processing - sensory input analysis
    Perception,
    /// Memory system - pattern matching and retrieval
    Memory,
    /// Executive control - goal-directed behavior
    Executive,
    /// Emotional valence - affective significance
    Emotion,
    /// Motor planning - action preparation
    Motor,
    /// Language - symbolic processing
    Language,
    /// Metacognition - self-monitoring
    Metacognition,
}

impl WorkspaceProcessor {
    pub fn all() -> [WorkspaceProcessor; 7] {
        [
            WorkspaceProcessor::Perception,
            WorkspaceProcessor::Memory,
            WorkspaceProcessor::Executive,
            WorkspaceProcessor::Emotion,
            WorkspaceProcessor::Motor,
            WorkspaceProcessor::Language,
            WorkspaceProcessor::Metacognition,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            WorkspaceProcessor::Perception => 0,
            WorkspaceProcessor::Memory => 1,
            WorkspaceProcessor::Executive => 2,
            WorkspaceProcessor::Emotion => 3,
            WorkspaceProcessor::Motor => 4,
            WorkspaceProcessor::Language => 5,
            WorkspaceProcessor::Metacognition => 6,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            WorkspaceProcessor::Perception => "Perception",
            WorkspaceProcessor::Memory => "Memory",
            WorkspaceProcessor::Executive => "Executive",
            WorkspaceProcessor::Emotion => "Emotion",
            WorkspaceProcessor::Motor => "Motor",
            WorkspaceProcessor::Language => "Language",
            WorkspaceProcessor::Metacognition => "Metacognition",
        }
    }
}

/// State of an individual processor competing for workspace access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorState {
    /// Processor type
    pub processor: WorkspaceProcessor,
    /// Current activation level [0, 1]
    pub activation: f64,
    /// Coalition strength (how well it's allied with others) [0, 1]
    pub coalition_strength: f64,
    /// Content signal strength (what it's trying to broadcast) [0, 1]
    pub content_strength: f64,
    /// Whether currently in conscious broadcast
    pub is_broadcasting: bool,
    /// Time since last broadcast (for refractory period)
    pub time_since_broadcast: f64,
}

impl ProcessorState {
    pub fn new(processor: WorkspaceProcessor) -> Self {
        Self {
            processor,
            activation: 0.3 + 0.1 * (processor.index() as f64 / 7.0), // Slight variation
            coalition_strength: 0.0,
            content_strength: 0.0,
            is_broadcasting: false,
            time_since_broadcast: 1.0, // Ready to broadcast
        }
    }

    /// Update activation based on inputs and neuromodulators
    pub fn update_activation(&mut self, input: f64, da: f64, ne: f64, dt: f64) {
        // Dopamine enhances activation for reward-relevant content
        let da_boost = 1.0 + 0.5 * da;
        // Norepinephrine affects overall gain
        let ne_gain = 0.5 + ne;

        // Leaky integration with neuromodulator modulation
        let target = (input * da_boost).min(1.0);
        let decay = 0.1 * ne_gain;

        self.activation += (target - self.activation) * decay * dt * 50.0;
        self.activation = self.activation.clamp(0.0, 1.0);

        // Update time since broadcast
        if !self.is_broadcasting {
            self.time_since_broadcast += dt;
        }
    }

    /// Compute effective competition strength
    pub fn competition_strength(&self) -> f64 {
        // Combined activation, coalition support, and content strength
        // Refractory period reduces strength after recent broadcast
        let refractory = (self.time_since_broadcast / 0.5).min(1.0); // 500ms refractory
        (self.activation * 0.4 + self.coalition_strength * 0.3 + self.content_strength * 0.3) * refractory
    }
}

/// Coalition of processors that can amplify each other's signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorCoalition {
    /// Primary processor leading the coalition
    pub leader: WorkspaceProcessor,
    /// Supporting processors in the coalition
    pub members: Vec<WorkspaceProcessor>,
    /// Combined strength of the coalition [0, 1]
    pub strength: f64,
    /// Coherence of coalition members [0, 1]
    pub coherence: f64,
}

/// Global Workspace state tracking competition, ignition, and broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspaceState {
    /// Individual processor states
    pub processors: Vec<ProcessorState>,

    /// Current ignition threshold (modulated by neuromodulators)
    pub ignition_threshold: f64,

    /// Whether workspace is currently in ignition state
    pub is_ignited: bool,

    /// Currently broadcasting processor (if ignited)
    pub broadcasting_processor: Option<WorkspaceProcessor>,

    /// Broadcast strength (gamma synchronization level) [0, 1]
    pub broadcast_strength: f64,

    /// Time since last ignition (for timing dynamics)
    pub time_since_ignition: f64,

    /// Duration of current broadcast
    pub broadcast_duration: f64,

    /// Maximum broadcast duration before fade
    pub max_broadcast_duration: f64,

    /// Current active coalitions
    pub active_coalitions: Vec<ProcessorCoalition>,

    /// Global workspace capacity (limited to 1 main + ~2-3 secondary)
    pub workspace_capacity: usize,

    /// Integration measure (how unified is the broadcast) [0, 1]
    pub integration: f64,

    /// Conscious access events counter
    pub ignition_count: u64,

    /// Total broadcast time
    pub total_broadcast_time: f64,

    /// Update counter
    pub updates: u64,
}

impl GlobalWorkspaceState {
    pub fn new() -> Self {
        let processors = WorkspaceProcessor::all()
            .iter()
            .map(|&p| ProcessorState::new(p))
            .collect();

        Self {
            processors,
            ignition_threshold: 0.6, // Base threshold
            is_ignited: false,
            broadcasting_processor: None,
            broadcast_strength: 0.0,
            time_since_ignition: 1.0,
            broadcast_duration: 0.0,
            max_broadcast_duration: 0.3, // 300ms typical conscious moment
            active_coalitions: Vec::new(),
            workspace_capacity: 4, // 1 main + 3 secondary (magical number 4)
            integration: 0.0,
            ignition_count: 0,
            total_broadcast_time: 0.0,
            updates: 0,
        }
    }

    /// Update processor activations based on oscillatory state
    pub fn update_processors(
        &mut self,
        phi: f64,
        attention_intensity: f64,
        gamma_power: f64,
        theta_gamma_coupling: f64,
        da: f64,
        ne: f64,
        surprise: f64,
        dt: f64,
    ) {
        for processor in &mut self.processors {
            // Each processor responds to different aspects of the state
            let input = match processor.processor {
                WorkspaceProcessor::Perception => {
                    // Perception responds to sensory clarity (attention + gamma)
                    attention_intensity * 0.6 + gamma_power * 0.4
                }
                WorkspaceProcessor::Memory => {
                    // Memory responds to theta-gamma coupling (memory encoding)
                    theta_gamma_coupling * 0.7 + phi * 0.3
                }
                WorkspaceProcessor::Executive => {
                    // Executive responds to integration (phi) and attention
                    phi * 0.5 + attention_intensity * 0.5
                }
                WorkspaceProcessor::Emotion => {
                    // Emotion responds to dopamine and surprise
                    da * 0.5 + surprise * 0.5
                }
                WorkspaceProcessor::Motor => {
                    // Motor responds to readiness (beta power proxy = 1 - gamma)
                    (1.0 - gamma_power) * 0.5 + attention_intensity * 0.5
                }
                WorkspaceProcessor::Language => {
                    // Language responds to gamma and integration
                    gamma_power * 0.4 + phi * 0.6
                }
                WorkspaceProcessor::Metacognition => {
                    // Metacognition responds to overall coherence
                    phi * 0.4 + theta_gamma_coupling * 0.3 + attention_intensity * 0.3
                }
            };

            processor.update_activation(input, da, ne, dt);

            // Update content strength based on activation history
            processor.content_strength = processor.activation * 0.8 + processor.content_strength * 0.2;
        }
    }

    /// Detect and update coalitions between processors
    pub fn update_coalitions(&mut self, gamma_coherence: f64) {
        self.active_coalitions.clear();

        // Find high-activation processors
        let active: Vec<_> = self.processors.iter()
            .filter(|p| p.activation > 0.4)
            .collect();

        if active.len() < 2 {
            return;
        }

        // Processors with similar high activation can form coalitions
        // Gamma coherence determines coalition strength
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let activation_similarity = 1.0 - (active[i].activation - active[j].activation).abs();
                if activation_similarity > 0.7 {
                    // These can form a coalition
                    let coalition = ProcessorCoalition {
                        leader: active[i].processor,
                        members: vec![active[j].processor],
                        strength: (active[i].activation + active[j].activation) / 2.0 * gamma_coherence,
                        coherence: gamma_coherence * activation_similarity,
                    };
                    self.active_coalitions.push(coalition);
                }
            }
        }

        // Update coalition strength for each processor
        for processor in &mut self.processors {
            processor.coalition_strength = self.active_coalitions.iter()
                .filter(|c| c.leader == processor.processor || c.members.contains(&processor.processor))
                .map(|c| c.strength)
                .sum::<f64>()
                .min(1.0);
        }
    }

    /// Check for ignition and handle broadcast dynamics
    pub fn update_ignition(
        &mut self,
        gamma_power: f64,
        gamma_coherence: f64,
        da: f64,
        serotonin: f64,
        dt: f64,
    ) {
        // Neuromodulators affect ignition threshold
        // High dopamine LOWERS threshold (easier ignition for rewarding stimuli)
        // High serotonin RAISES threshold (more stable, less impulsive)
        self.ignition_threshold = (0.6 - 0.15 * da + 0.1 * serotonin).clamp(0.3, 0.8);

        if self.is_ignited {
            // Currently broadcasting - update duration and check for fade
            self.broadcast_duration += dt;
            self.broadcast_strength = gamma_power * gamma_coherence;

            // Update integration based on how unified the broadcast is
            self.integration = gamma_coherence * (1.0 - self.broadcast_duration / self.max_broadcast_duration).max(0.0);

            // Check if broadcast should end
            if self.broadcast_duration > self.max_broadcast_duration || self.broadcast_strength < 0.3 {
                // End broadcast
                self.is_ignited = false;
                if let Some(processor) = self.broadcasting_processor {
                    if let Some(p) = self.processors.iter_mut().find(|p| p.processor == processor) {
                        p.is_broadcasting = false;
                        p.time_since_broadcast = 0.0;
                    }
                }
                self.broadcasting_processor = None;
                self.broadcast_duration = 0.0;
            }

            self.total_broadcast_time += dt;
        } else {
            // Not currently broadcasting - check for new ignition
            self.time_since_ignition += dt;

            // Need minimum time between ignitions (refractory period)
            if self.time_since_ignition < 0.1 {
                return;
            }

            // Find processor with highest competition strength
            let winner = self.processors.iter()
                .max_by(|a, b| a.competition_strength().partial_cmp(&b.competition_strength()).unwrap());

            if let Some(winner) = winner {
                if winner.competition_strength() > self.ignition_threshold {
                    // IGNITION! This processor wins access to consciousness
                    self.is_ignited = true;
                    self.broadcasting_processor = Some(winner.processor);
                    self.broadcast_strength = gamma_power * gamma_coherence;
                    self.broadcast_duration = 0.0;
                    self.time_since_ignition = 0.0;
                    self.ignition_count += 1;
                    self.integration = gamma_coherence;

                    // Mark the winner as broadcasting
                    if let Some(p) = self.processors.iter_mut().find(|p| p.processor == winner.processor) {
                        p.is_broadcasting = true;
                    }
                }
            }
        }

        self.updates += 1;
    }

    /// Get the dominant (highest activation) processor
    pub fn dominant_processor(&self) -> WorkspaceProcessor {
        self.processors.iter()
            .max_by(|a, b| a.activation.partial_cmp(&b.activation).unwrap())
            .map(|p| p.processor)
            .unwrap_or(WorkspaceProcessor::Perception)
    }

    /// Get activation for a specific processor
    pub fn processor_activation(&self, processor: WorkspaceProcessor) -> f64 {
        self.processors.iter()
            .find(|p| p.processor == processor)
            .map(|p| p.activation)
            .unwrap_or(0.0)
    }

    /// Check if a specific processor is currently broadcasting
    pub fn is_processor_broadcasting(&self, processor: WorkspaceProcessor) -> bool {
        self.broadcasting_processor == Some(processor)
    }

    /// Get summary for MindState integration
    pub fn summary(&self) -> GlobalWorkspaceSummary {
        GlobalWorkspaceSummary {
            is_ignited: self.is_ignited,
            broadcasting_processor: self.broadcasting_processor.map(|p| p.name().to_string()),
            broadcast_strength: self.broadcast_strength,
            broadcast_duration: self.broadcast_duration,
            ignition_threshold: self.ignition_threshold,
            time_since_ignition: self.time_since_ignition,
            integration: self.integration,
            coalition_count: self.active_coalitions.len(),
            perception_activation: self.processor_activation(WorkspaceProcessor::Perception),
            memory_activation: self.processor_activation(WorkspaceProcessor::Memory),
            executive_activation: self.processor_activation(WorkspaceProcessor::Executive),
            emotion_activation: self.processor_activation(WorkspaceProcessor::Emotion),
            motor_activation: self.processor_activation(WorkspaceProcessor::Motor),
            language_activation: self.processor_activation(WorkspaceProcessor::Language),
            metacognition_activation: self.processor_activation(WorkspaceProcessor::Metacognition),
            ignition_count: self.ignition_count,
            total_broadcast_time: self.total_broadcast_time,
            updates: self.updates,
        }
    }
}

impl Default for GlobalWorkspaceState {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of global workspace state for MindState integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspaceSummary {
    /// Whether workspace is currently ignited (conscious access)
    pub is_ignited: bool,
    /// Currently broadcasting processor (if ignited)
    pub broadcasting_processor: Option<String>,
    /// Broadcast strength [0, 1]
    pub broadcast_strength: f64,
    /// Current broadcast duration in seconds
    pub broadcast_duration: f64,
    /// Current ignition threshold
    pub ignition_threshold: f64,
    /// Time since last ignition
    pub time_since_ignition: f64,
    /// Integration level [0, 1]
    pub integration: f64,
    /// Number of active coalitions
    pub coalition_count: usize,
    /// Processor activations
    pub perception_activation: f64,
    pub memory_activation: f64,
    pub executive_activation: f64,
    pub emotion_activation: f64,
    pub motor_activation: f64,
    pub language_activation: f64,
    pub metacognition_activation: f64,
    /// Statistics
    pub ignition_count: u64,
    pub total_broadcast_time: f64,
    pub updates: u64,
}

/// State of a single oscillatory band
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandState {
    /// Current phase [0, 2π]
    pub phase: f64,
    /// Current frequency within band range
    pub frequency: f64,
    /// Power/amplitude of this band [0, 1]
    pub power: f64,
    /// Coherence within this band [0, 1]
    pub coherence: f64,
}

impl BandState {
    pub fn new(band: FrequencyBand) -> Self {
        Self {
            phase: 0.0,
            frequency: band.center_frequency(),
            power: 0.5,
            coherence: 0.5,
        }
    }

    /// Advance phase by dt seconds
    pub fn advance(&mut self, dt: f64) {
        self.phase = (self.phase + 2.0 * PI * self.frequency * dt) % (2.0 * PI);
    }
}

/// Multi-band oscillatory state tracking all frequency bands simultaneously
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBandState {
    /// Individual band states
    pub bands: HashMap<FrequencyBand, BandState>,
    /// Current dominant band
    pub dominant_band: FrequencyBand,
    /// Cross-frequency coupling strength (theta-gamma)
    pub theta_gamma_coupling: f64,
    /// Cross-frequency coupling strength (alpha-gamma, inhibitory)
    pub alpha_gamma_coupling: f64,
    /// Overall arousal level [0, 1]
    pub arousal: f64,
    /// Attention focus level [0, 1]
    pub attention: f64,
}

impl MultiBandState {
    pub fn new() -> Self {
        let mut bands = HashMap::new();
        // Phase 7: Add sleep-specific bands (initially low power when awake)
        bands.insert(FrequencyBand::Delta, {
            let mut state = BandState::new(FrequencyBand::Delta);
            state.power = 0.1; // Low during wakefulness
            state
        });
        bands.insert(FrequencyBand::SleepSpindle, {
            let mut state = BandState::new(FrequencyBand::SleepSpindle);
            state.power = 0.1; // Low during wakefulness
            state
        });
        // Original wake-state bands
        bands.insert(FrequencyBand::Theta, BandState::new(FrequencyBand::Theta));
        bands.insert(FrequencyBand::Alpha, BandState::new(FrequencyBand::Alpha));
        bands.insert(FrequencyBand::Beta, BandState::new(FrequencyBand::Beta));
        bands.insert(FrequencyBand::Gamma, BandState::new(FrequencyBand::Gamma));

        Self {
            bands,
            dominant_band: FrequencyBand::Gamma,
            theta_gamma_coupling: 0.5,
            alpha_gamma_coupling: 0.3,
            arousal: 0.6,
            attention: 0.5,
        }
    }

    /// Advance all bands by dt seconds with cross-frequency coupling
    pub fn advance(&mut self, dt: f64) {
        // Advance each band
        for band_state in self.bands.values_mut() {
            band_state.advance(dt);
        }

        // ===== THETA-GAMMA COUPLING =====
        // Gamma bursts are nested within theta cycles
        // When theta is at peak, gamma power increases (phase-amplitude coupling)
        if let (Some(theta), Some(gamma)) = (
            self.bands.get(&FrequencyBand::Theta),
            self.bands.get_mut(&FrequencyBand::Gamma),
        ) {
            let theta_phase_factor = (theta.phase.cos() + 1.0) / 2.0; // 0-1 based on theta phase
            let coupling_effect = self.theta_gamma_coupling * theta_phase_factor * 0.3;
            gamma.power = (gamma.power + coupling_effect).min(1.0);
        }

        // ===== ALPHA-GAMMA COUPLING (inhibitory) =====
        // High alpha power suppresses gamma (attention gating)
        if let (Some(alpha), Some(gamma)) = (
            self.bands.get(&FrequencyBand::Alpha),
            self.bands.get_mut(&FrequencyBand::Gamma),
        ) {
            let suppression = self.alpha_gamma_coupling * alpha.power * 0.2;
            gamma.power = (gamma.power - suppression).max(0.1);
        }
    }

    /// Update arousal level and shift dominant band accordingly
    pub fn set_arousal(&mut self, arousal: f64) {
        self.arousal = arousal.clamp(0.0, 1.0);

        // Arousal shifts power distribution across bands
        // Low arousal → Alpha/Theta dominant
        // High arousal → Beta/Gamma dominant
        for (band, state) in self.bands.iter_mut() {
            let optimal = band.optimal_arousal();
            let distance = (self.arousal - optimal).abs();
            let power_factor = 1.0 - distance;
            state.power = state.power * 0.7 + power_factor * 0.3;
        }

        // Update dominant band
        self.dominant_band = self.get_dominant_band();
    }

    /// Set attention level (modulates gamma and suppresses alpha)
    pub fn set_attention(&mut self, attention: f64) {
        self.attention = attention.clamp(0.0, 1.0);

        // High attention boosts gamma, suppresses alpha
        if let Some(gamma) = self.bands.get_mut(&FrequencyBand::Gamma) {
            gamma.power = (gamma.power + attention * 0.2).min(1.0);
            gamma.coherence = (gamma.coherence + attention * 0.1).min(1.0);
        }
        if let Some(alpha) = self.bands.get_mut(&FrequencyBand::Alpha) {
            alpha.power = (alpha.power - attention * 0.3).max(0.1);
        }
    }

    /// Get the currently dominant band based on power
    pub fn get_dominant_band(&self) -> FrequencyBand {
        self.bands
            .iter()
            .max_by(|a, b| a.1.power.partial_cmp(&b.1.power).unwrap())
            .map(|(band, _)| *band)
            .unwrap_or(FrequencyBand::Gamma)
    }

    /// Get gamma phase (primary consciousness oscillation)
    pub fn gamma_phase(&self) -> f64 {
        self.bands.get(&FrequencyBand::Gamma).map(|s| s.phase).unwrap_or(0.0)
    }

    /// Get theta phase (memory/navigation oscillation)
    pub fn theta_phase(&self) -> f64 {
        self.bands.get(&FrequencyBand::Theta).map(|s| s.phase).unwrap_or(0.0)
    }

    /// Get combined power-weighted coherence
    pub fn overall_coherence(&self) -> f64 {
        let total_power: f64 = self.bands.values().map(|s| s.power).sum();
        if total_power < 0.01 {
            return 0.5;
        }
        self.bands.values()
            .map(|s| s.coherence * s.power)
            .sum::<f64>() / total_power
    }

    /// Check if in theta-gamma coupling sweet spot (memory encoding)
    pub fn in_memory_encoding_window(&self) -> bool {
        if let (Some(theta), Some(gamma)) = (
            self.bands.get(&FrequencyBand::Theta),
            self.bands.get(&FrequencyBand::Gamma),
        ) {
            // Memory encoding optimal when theta at peak and gamma power high
            let theta_at_peak = theta.phase.cos() > 0.7;
            let gamma_strong = gamma.power > 0.6;
            theta_at_peak && gamma_strong
        } else {
            false
        }
    }

    /// Get summary for external monitoring
    pub fn summary(&self) -> MultiBandSummary {
        MultiBandSummary {
            dominant_band: self.dominant_band,
            gamma_power: self.bands.get(&FrequencyBand::Gamma).map(|s| s.power).unwrap_or(0.0),
            theta_power: self.bands.get(&FrequencyBand::Theta).map(|s| s.power).unwrap_or(0.0),
            alpha_power: self.bands.get(&FrequencyBand::Alpha).map(|s| s.power).unwrap_or(0.0),
            beta_power: self.bands.get(&FrequencyBand::Beta).map(|s| s.power).unwrap_or(0.0),
            theta_gamma_coupling: self.theta_gamma_coupling,
            overall_coherence: self.overall_coherence(),
            arousal: self.arousal,
            attention: self.attention,
            in_memory_window: self.in_memory_encoding_window(),
        }
    }
}

impl Default for MultiBandState {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of multi-band oscillatory state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBandSummary {
    pub dominant_band: FrequencyBand,
    pub gamma_power: f64,
    pub theta_power: f64,
    pub alpha_power: f64,
    pub beta_power: f64,
    pub theta_gamma_coupling: f64,
    pub overall_coherence: f64,
    pub arousal: f64,
    pub attention: f64,
    pub in_memory_window: bool,
}

/// Type of memory operation for oscillatory gating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOperationType {
    /// Encoding new memories (optimal during theta-gamma coupling)
    Encode,
    /// Retrieving existing memories (optimal during high gamma)
    Retrieve,
    /// Consolidating memories (optimal during low arousal)
    Consolidate,
    /// No memory operation (baseline state)
    Idle,
}

// ═══════════════════════════════════════════════════════════════════════════
// PROCESS RESONANCE DETECTION (Phase 6C)
// ═══════════════════════════════════════════════════════════════════════════
//
// Detects when multiple cognitive processes achieve phase-locked synchrony.
// Based on neural binding theory: synchronized oscillations bind distributed
// representations into unified conscious experiences.
//
// Research Foundation:
// - Binding by synchrony (Singer & Gray, 1995)
// - Phase-locking value (PLV) for neural synchrony (Lachaux et al., 1999)
// - Communication through coherence (Fries, 2005, 2015)
// - Gamma-band binding in perception (Engel & Singer, 2001)
// ═══════════════════════════════════════════════════════════════════════════

/// Unique identifier for a cognitive process
pub type ProcessId = u64;

/// State of a single cognitive process for resonance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessOscillation {
    /// Process identifier
    pub id: ProcessId,
    /// Process name/type
    pub name: String,
    /// Current phase [0, 2π]
    pub phase: f64,
    /// Natural frequency (Hz)
    pub frequency: f64,
    /// Amplitude/strength [0, 1]
    pub amplitude: f64,
    /// Last update timestamp
    pub last_update: f64,
    /// Is this process currently active?
    pub active: bool,
}

impl ProcessOscillation {
    pub fn new(id: ProcessId, name: &str, frequency: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            phase: 0.0,
            frequency,
            amplitude: 0.5,
            last_update: 0.0,
            active: true,
        }
    }

    /// Advance phase by dt seconds
    pub fn advance(&mut self, dt: f64) {
        self.phase = (self.phase + 2.0 * PI * self.frequency * dt) % (2.0 * PI);
        self.last_update += dt;
    }

    /// Calculate phase difference with another process [-π, π]
    pub fn phase_difference(&self, other: &ProcessOscillation) -> f64 {
        let diff = self.phase - other.phase;
        // Normalize to [-π, π]
        ((diff + PI) % (2.0 * PI)) - PI
    }
}

/// Resonance state between two processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessResonance {
    /// First process ID
    pub process_a: ProcessId,
    /// Second process ID
    pub process_b: ProcessId,
    /// Phase-Locking Value (PLV) [0, 1] - 1 = perfect synchrony
    pub plv: f64,
    /// Mean phase difference (preferred phase relationship)
    pub mean_phase_diff: f64,
    /// Coherence strength [0, 1]
    pub coherence: f64,
    /// Number of samples used for PLV calculation
    pub sample_count: usize,
    /// Is this pair currently resonating (PLV > threshold)?
    pub is_resonating: bool,
}

impl ProcessResonance {
    pub fn new(process_a: ProcessId, process_b: ProcessId) -> Self {
        Self {
            process_a,
            process_b,
            plv: 0.0,
            mean_phase_diff: 0.0,
            coherence: 0.0,
            sample_count: 0,
            is_resonating: false,
        }
    }
}

/// A cluster of mutually resonating processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantCluster {
    /// Process IDs in this cluster
    pub processes: Vec<ProcessId>,
    /// Mean PLV across all pairs in cluster
    pub mean_plv: f64,
    /// Cluster coherence (how tightly bound)
    pub coherence: f64,
    /// Dominant frequency of the cluster
    pub dominant_frequency: f64,
    /// Integration potential (higher = more conscious binding)
    pub integration_potential: f64,
}

/// Process Resonance Detector
///
/// Tracks multiple cognitive processes and detects phase-locked synchrony
/// between them. Identifies resonant clusters for enhanced binding.
#[derive(Debug, Clone)]
pub struct ResonanceDetector {
    /// Tracked processes
    processes: HashMap<ProcessId, ProcessOscillation>,
    /// Pairwise resonance states
    resonances: HashMap<(ProcessId, ProcessId), ProcessResonance>,
    /// Phase difference history for PLV calculation
    phase_histories: HashMap<(ProcessId, ProcessId), VecDeque<f64>>,
    /// PLV threshold for resonance detection
    plv_threshold: f64,
    /// History length for PLV calculation
    history_length: usize,
    /// Next process ID
    next_id: ProcessId,
    /// Detected resonant clusters
    clusters: Vec<ResonantCluster>,
}

impl ResonanceDetector {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            resonances: HashMap::new(),
            phase_histories: HashMap::new(),
            plv_threshold: 0.7,  // PLV > 0.7 indicates significant synchrony
            history_length: 50,  // ~50 samples for stable PLV estimate
            next_id: 0,
            clusters: Vec::new(),
        }
    }

    /// Register a new process for tracking
    pub fn register_process(&mut self, name: &str, frequency: f64) -> ProcessId {
        let id = self.next_id;
        self.next_id += 1;

        let process = ProcessOscillation::new(id, name, frequency);
        self.processes.insert(id, process);

        // Initialize resonance tracking with all existing processes
        for &existing_id in self.processes.keys() {
            if existing_id != id {
                let key = if existing_id < id { (existing_id, id) } else { (id, existing_id) };
                self.resonances.insert(key, ProcessResonance::new(key.0, key.1));
                self.phase_histories.insert(key, VecDeque::with_capacity(self.history_length));
            }
        }

        id
    }

    /// Update a process's oscillatory state
    pub fn update_process(&mut self, id: ProcessId, phase: f64, amplitude: f64, active: bool) {
        if let Some(process) = self.processes.get_mut(&id) {
            process.phase = phase;
            process.amplitude = amplitude;
            process.active = active;
        }
    }

    /// Advance all processes and update resonance calculations
    pub fn advance(&mut self, dt: f64) {
        // Advance all process phases
        for process in self.processes.values_mut() {
            if process.active {
                process.advance(dt);
            }
        }

        // Update pairwise PLV calculations
        let process_ids: Vec<ProcessId> = self.processes.keys().copied().collect();

        for i in 0..process_ids.len() {
            for j in (i + 1)..process_ids.len() {
                let id_a = process_ids[i];
                let id_b = process_ids[j];
                let key = (id_a, id_b);

                if let (Some(proc_a), Some(proc_b)) = (
                    self.processes.get(&id_a),
                    self.processes.get(&id_b),
                ) {
                    // Only track if both active
                    if proc_a.active && proc_b.active {
                        let phase_diff = proc_a.phase_difference(proc_b);

                        // Add to history
                        if let Some(history) = self.phase_histories.get_mut(&key) {
                            history.push_back(phase_diff);
                            while history.len() > self.history_length {
                                history.pop_front();
                            }

                            // Calculate PLV from history
                            if history.len() >= 10 {
                                let plv = self.calculate_plv(history);
                                let mean_diff = self.calculate_mean_phase(history);

                                if let Some(resonance) = self.resonances.get_mut(&key) {
                                    resonance.plv = plv;
                                    resonance.mean_phase_diff = mean_diff;
                                    resonance.coherence = plv * (proc_a.amplitude * proc_b.amplitude).sqrt();
                                    resonance.sample_count = history.len();
                                    resonance.is_resonating = plv > self.plv_threshold;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Detect clusters
        self.detect_clusters();
    }

    /// Calculate Phase-Locking Value from phase difference history
    /// PLV = |mean(exp(i * phase_diff))|
    fn calculate_plv(&self, history: &VecDeque<f64>) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        // PLV = magnitude of mean unit vector
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for &phase_diff in history {
            sum_cos += phase_diff.cos();
            sum_sin += phase_diff.sin();
        }

        let n = history.len() as f64;
        let mean_cos = sum_cos / n;
        let mean_sin = sum_sin / n;

        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }

    /// Calculate mean phase difference (circular mean)
    fn calculate_mean_phase(&self, history: &VecDeque<f64>) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for &phase_diff in history {
            sum_cos += phase_diff.cos();
            sum_sin += phase_diff.sin();
        }

        sum_sin.atan2(sum_cos)
    }

    /// Detect clusters of mutually resonating processes
    fn detect_clusters(&mut self) {
        self.clusters.clear();

        // Find all resonating pairs
        let resonating_pairs: Vec<(ProcessId, ProcessId)> = self.resonances
            .iter()
            .filter(|(_, r)| r.is_resonating)
            .map(|(&k, _)| k)
            .collect();

        if resonating_pairs.is_empty() {
            return;
        }

        // Build adjacency for cluster detection
        let mut visited: HashMap<ProcessId, bool> = HashMap::new();
        for &id in self.processes.keys() {
            visited.insert(id, false);
        }

        // Find connected components (clusters)
        for &start_id in self.processes.keys() {
            if visited.get(&start_id) == Some(&true) {
                continue;
            }

            let mut cluster_ids = Vec::new();
            let mut stack = vec![start_id];

            while let Some(current) = stack.pop() {
                if visited.get(&current) == Some(&true) {
                    continue;
                }

                visited.insert(current, true);
                cluster_ids.push(current);

                // Find neighbors through resonating pairs
                for &(a, b) in &resonating_pairs {
                    if a == current && visited.get(&b) != Some(&true) {
                        stack.push(b);
                    } else if b == current && visited.get(&a) != Some(&true) {
                        stack.push(a);
                    }
                }
            }

            // Only create cluster if >1 process
            if cluster_ids.len() > 1 {
                let cluster = self.build_cluster(&cluster_ids);
                self.clusters.push(cluster);
            }
        }

        // Sort clusters by integration potential
        self.clusters.sort_by(|a, b|
            b.integration_potential.partial_cmp(&a.integration_potential).unwrap()
        );
    }

    /// Build a ResonantCluster from process IDs
    fn build_cluster(&self, process_ids: &[ProcessId]) -> ResonantCluster {
        let n = process_ids.len();

        // Calculate mean PLV across cluster
        let mut total_plv = 0.0;
        let mut pair_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let key = if process_ids[i] < process_ids[j] {
                    (process_ids[i], process_ids[j])
                } else {
                    (process_ids[j], process_ids[i])
                };

                if let Some(resonance) = self.resonances.get(&key) {
                    total_plv += resonance.plv;
                    pair_count += 1;
                }
            }
        }

        let mean_plv = if pair_count > 0 { total_plv / pair_count as f64 } else { 0.0 };

        // Calculate dominant frequency (weighted by amplitude)
        let mut total_freq = 0.0;
        let mut total_amp = 0.0;

        for &id in process_ids {
            if let Some(proc) = self.processes.get(&id) {
                total_freq += proc.frequency * proc.amplitude;
                total_amp += proc.amplitude;
            }
        }

        let dominant_freq = if total_amp > 0.0 { total_freq / total_amp } else { 40.0 };

        // Cluster coherence (how tight the binding)
        let coherence = mean_plv * (n as f64).sqrt() / (n as f64);

        // Integration potential: PLV × cluster size × coherence
        // Larger synchronized clusters = higher integration
        let integration_potential = mean_plv * (n as f64).ln().max(1.0) * coherence;

        ResonantCluster {
            processes: process_ids.to_vec(),
            mean_plv,
            coherence,
            dominant_frequency: dominant_freq,
            integration_potential,
        }
    }

    /// Get all currently resonating process pairs
    pub fn get_resonating_pairs(&self) -> Vec<&ProcessResonance> {
        self.resonances.values().filter(|r| r.is_resonating).collect()
    }

    /// Get detected resonant clusters
    pub fn get_clusters(&self) -> &[ResonantCluster] {
        &self.clusters
    }

    /// Get the largest/most integrated cluster
    pub fn get_primary_cluster(&self) -> Option<&ResonantCluster> {
        self.clusters.first()
    }

    /// Get overall system resonance (mean PLV across all active pairs)
    pub fn system_resonance(&self) -> f64 {
        let active_resonances: Vec<f64> = self.resonances
            .values()
            .filter(|r| r.sample_count > 10)
            .map(|r| r.plv)
            .collect();

        if active_resonances.is_empty() {
            return 0.0;
        }

        active_resonances.iter().sum::<f64>() / active_resonances.len() as f64
    }

    /// Get integration boost factor based on resonance state
    /// High resonance → enhanced binding → higher effective Φ
    pub fn integration_boost(&self) -> f64 {
        let sys_res = self.system_resonance();
        let cluster_boost = self.clusters.first()
            .map(|c| c.integration_potential)
            .unwrap_or(0.0);

        // Base boost from system resonance (up to 20%)
        let base_boost = sys_res * 0.2;

        // Additional boost from primary cluster (up to 15%)
        let cluster_contribution = cluster_boost * 0.15;

        1.0 + base_boost + cluster_contribution
    }

    /// Get summary for monitoring
    pub fn summary(&self) -> ResonanceDetectorSummary {
        ResonanceDetectorSummary {
            process_count: self.processes.len(),
            active_processes: self.processes.values().filter(|p| p.active).count(),
            resonating_pairs: self.resonances.values().filter(|r| r.is_resonating).count(),
            cluster_count: self.clusters.len(),
            system_resonance: self.system_resonance(),
            integration_boost: self.integration_boost(),
            primary_cluster_size: self.clusters.first().map(|c| c.processes.len()).unwrap_or(0),
            primary_cluster_plv: self.clusters.first().map(|c| c.mean_plv).unwrap_or(0.0),
        }
    }
}

impl Default for ResonanceDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of resonance detector state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceDetectorSummary {
    pub process_count: usize,
    pub active_processes: usize,
    pub resonating_pairs: usize,
    pub cluster_count: usize,
    pub system_resonance: f64,
    pub integration_boost: f64,
    pub primary_cluster_size: usize,
    pub primary_cluster_plv: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION SPOTLIGHT SYSTEM (Phase 6C)
// ═══════════════════════════════════════════════════════════════════════════
//
// Implements selective attention as a "spotlight" that enhances processing
// for attended processes while suppressing non-attended ones.
//
// Research Foundation:
// - Spotlight theory of attention (Posner, 1980)
// - Biased competition model (Desimone & Duncan, 1995)
// - Selective attention through gamma enhancement (Fries et al., 2001)
// - Attention as precision weighting (Feldman & Friston, 2010)
// - Rhythmic sampling theory (VanRullen, 2016)
// ═══════════════════════════════════════════════════════════════════════════

/// A focus of attention (single spotlight)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFocus {
    /// Target process ID (-1 for no specific target)
    pub target_process: Option<ProcessId>,
    /// Target position in abstract "attention space" [0, 1]
    pub position: f64,
    /// Spotlight width (narrow = focused, wide = diffuse) [0.1, 1.0]
    pub width: f64,
    /// Intensity of attention [0, 1]
    pub intensity: f64,
    /// Duration the focus has been held (seconds)
    pub hold_duration: f64,
    /// Decay rate when not refreshed
    pub decay_rate: f64,
}

impl AttentionFocus {
    pub fn new(position: f64, intensity: f64) -> Self {
        Self {
            target_process: None,
            position: position.clamp(0.0, 1.0),
            width: 0.3,  // Default moderate focus
            intensity: intensity.clamp(0.0, 1.0),
            hold_duration: 0.0,
            decay_rate: 0.1,  // Lose 10% intensity per second without refresh
        }
    }

    /// Create a focus targeted at a specific process
    pub fn on_process(process_id: ProcessId, intensity: f64) -> Self {
        Self {
            target_process: Some(process_id),
            position: 0.5,  // Position less relevant for targeted attention
            width: 0.2,     // Narrow focus for specific target
            intensity: intensity.clamp(0.0, 1.0),
            hold_duration: 0.0,
            decay_rate: 0.05,  // Slower decay for specific targets
        }
    }

    /// Update focus over time
    pub fn update(&mut self, dt: f64, refreshed: bool) {
        if refreshed {
            self.hold_duration += dt;
            // Intensity increases slightly with sustained attention
            self.intensity = (self.intensity + 0.01 * dt).min(1.0);
        } else {
            // Decay without refresh
            self.intensity = (self.intensity - self.decay_rate * dt).max(0.0);
            if self.intensity < 0.1 {
                self.hold_duration = 0.0;  // Reset hold when attention wanes
            }
        }
    }

    /// Calculate enhancement factor for a process at given position
    /// Returns value [0, 1] where 0 = suppressed, 0.5 = neutral, 1 = enhanced
    pub fn enhancement_for(&self, process_position: f64) -> f64 {
        // Gaussian falloff from focus center
        let distance = (self.position - process_position).abs();
        let gaussian = (-distance * distance / (2.0 * self.width * self.width)).exp();

        // Enhancement = base + intensity-scaled gaussian
        // Non-attended areas get suppressed (factor < 0.5)
        let base_suppression = 0.3;  // Non-attended get 30% baseline
        let enhancement_boost = 0.7 * self.intensity * gaussian;

        base_suppression + enhancement_boost
    }

    /// Check if this focus targets a specific process
    pub fn targets(&self, process_id: ProcessId) -> bool {
        self.target_process == Some(process_id)
    }

    /// Is this focus still active (intensity > threshold)?
    pub fn is_active(&self) -> bool {
        self.intensity > 0.1
    }
}

/// Attention Spotlight System
///
/// Manages multiple attention foci (split attention support) and calculates
/// enhancement/suppression factors for cognitive processes.
#[derive(Debug, Clone)]
pub struct AttentionSpotlight {
    /// Primary attention focus
    primary_focus: AttentionFocus,
    /// Secondary foci (for split attention)
    secondary_foci: Vec<AttentionFocus>,
    /// Maximum number of simultaneous foci
    max_foci: usize,
    /// Global attention capacity [0, 1] (decreases with more foci)
    attention_capacity: f64,
    /// Process-specific enhancement factors (cached)
    process_enhancements: HashMap<ProcessId, f64>,
    /// Attentional blink state (refractory period after rapid switching)
    blink_remaining: f64,
    /// Spotlight oscillation phase (attention samples at ~7Hz theta)
    sampling_phase: f64,
    /// Sampling frequency (theta-band, ~7Hz)
    sampling_frequency: f64,
}

impl AttentionSpotlight {
    pub fn new() -> Self {
        Self {
            primary_focus: AttentionFocus::new(0.5, 0.5),  // Start centered, moderate
            secondary_foci: Vec::new(),
            max_foci: 4,  // Typical limit of 4±1 items
            attention_capacity: 1.0,
            process_enhancements: HashMap::new(),
            blink_remaining: 0.0,
            sampling_phase: 0.0,
            sampling_frequency: 7.0,  // Theta-band attention sampling
        }
    }

    /// Set primary attention focus
    pub fn focus_on(&mut self, position: f64, intensity: f64) {
        // Check for attentional blink
        if self.blink_remaining > 0.0 {
            return;  // Cannot shift during blink
        }

        let distance = (self.primary_focus.position - position).abs();

        // Large shifts trigger attentional blink
        if distance > 0.5 {
            self.blink_remaining = 0.1;  // 100ms blink
        }

        self.primary_focus = AttentionFocus::new(position, intensity);
    }

    /// Focus on a specific process
    pub fn focus_on_process(&mut self, process_id: ProcessId, intensity: f64) {
        if self.blink_remaining > 0.0 {
            return;
        }

        self.primary_focus = AttentionFocus::on_process(process_id, intensity);
    }

    /// Add secondary attention focus (split attention)
    pub fn add_secondary_focus(&mut self, focus: AttentionFocus) -> bool {
        if self.secondary_foci.len() >= self.max_foci - 1 {
            return false;  // Capacity reached
        }

        self.secondary_foci.push(focus);
        self.recalculate_capacity();
        true
    }

    /// Remove all secondary foci
    pub fn consolidate_attention(&mut self) {
        self.secondary_foci.clear();
        self.attention_capacity = 1.0;
    }

    /// Advance spotlight state
    pub fn advance(&mut self, dt: f64) {
        // Update attentional blink
        if self.blink_remaining > 0.0 {
            self.blink_remaining = (self.blink_remaining - dt).max(0.0);
        }

        // Update sampling phase (attention samples periodically)
        self.sampling_phase = (self.sampling_phase + 2.0 * PI * self.sampling_frequency * dt) % (2.0 * PI);

        // Update primary focus
        self.primary_focus.update(dt, true);  // Assume primary is always refreshed

        // Update and prune secondary foci
        self.secondary_foci.retain_mut(|focus| {
            focus.update(dt, false);  // Secondary foci decay unless explicitly refreshed
            focus.is_active()
        });

        self.recalculate_capacity();
    }

    /// Recalculate attention capacity based on number of foci
    fn recalculate_capacity(&mut self) {
        // Capacity decreases with more foci (1/sqrt(n) rule)
        let n_foci = 1.0 + self.secondary_foci.len() as f64;
        self.attention_capacity = 1.0 / n_foci.sqrt();
    }

    /// Get enhancement factor for a process
    /// Returns [0, 1] where 0.5 = neutral, >0.5 = enhanced, <0.5 = suppressed
    pub fn get_enhancement(&self, process_id: ProcessId, process_position: f64) -> f64 {
        // Check if in attentional blink
        if self.blink_remaining > 0.0 {
            return 0.3;  // Reduced processing during blink
        }

        // Check for direct process targeting
        if self.primary_focus.targets(process_id) {
            return 0.5 + 0.5 * self.primary_focus.intensity * self.attention_capacity;
        }

        for focus in &self.secondary_foci {
            if focus.targets(process_id) {
                return 0.5 + 0.3 * focus.intensity * self.attention_capacity;
            }
        }

        // Calculate spatial enhancement from all foci
        let mut total_enhancement = self.primary_focus.enhancement_for(process_position);

        for focus in &self.secondary_foci {
            let secondary_contribution = focus.enhancement_for(process_position) * 0.5;
            total_enhancement = total_enhancement.max(secondary_contribution);
        }

        // Apply capacity scaling
        let centered_enhancement = total_enhancement - 0.5;
        0.5 + centered_enhancement * self.attention_capacity
    }

    /// Get current sampling phase [0, 2π]
    /// Attention effectiveness varies with this phase (rhythmic sampling)
    pub fn sampling_phase(&self) -> f64 {
        self.sampling_phase
    }

    /// Get sampling effectiveness (based on phase)
    /// Peak attention at phase 0, trough at π
    pub fn sampling_effectiveness(&self) -> f64 {
        // Cosine modulation: 0.8 to 1.0 (20% rhythmic variation)
        0.9 + 0.1 * self.sampling_phase.cos()
    }

    /// Is attention currently in "uptake" phase (good for encoding)?
    pub fn in_uptake_phase(&self) -> bool {
        self.sampling_phase.cos() > 0.5
    }

    /// Get summary for monitoring
    pub fn summary(&self) -> AttentionSpotlightSummary {
        AttentionSpotlightSummary {
            primary_position: self.primary_focus.position,
            primary_intensity: self.primary_focus.intensity,
            primary_width: self.primary_focus.width,
            secondary_foci_count: self.secondary_foci.len(),
            attention_capacity: self.attention_capacity,
            in_blink: self.blink_remaining > 0.0,
            sampling_phase: self.sampling_phase,
            sampling_effectiveness: self.sampling_effectiveness(),
            in_uptake_phase: self.in_uptake_phase(),
        }
    }

    /// Refresh a secondary focus (prevent decay)
    pub fn refresh_secondary(&mut self, index: usize) {
        if let Some(focus) = self.secondary_foci.get_mut(index) {
            focus.intensity = (focus.intensity + 0.1).min(1.0);
        }
    }
}

impl Default for AttentionSpotlight {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of attention spotlight state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSpotlightSummary {
    /// Position of primary attention focus [0, 1]
    pub primary_position: f64,
    /// Intensity of primary focus [0, 1]
    pub primary_intensity: f64,
    /// Width of primary spotlight [0.1, 1.0]
    pub primary_width: f64,
    /// Number of active secondary foci
    pub secondary_foci_count: usize,
    /// Current attention capacity [0, 1]
    pub attention_capacity: f64,
    /// Is attention in "blink" (refractory) state?
    pub in_blink: bool,
    /// Current sampling phase [0, 2π]
    pub sampling_phase: f64,
    /// Current sampling effectiveness [0.8, 1.0]
    pub sampling_effectiveness: f64,
    /// Is attention in uptake (encoding) phase?
    pub in_uptake_phase: bool,
}

impl MemoryOperationType {
    /// Get the optimal frequency band for this operation
    pub fn optimal_band(&self) -> FrequencyBand {
        match self {
            MemoryOperationType::Encode => FrequencyBand::Theta, // Theta-gamma coupling
            MemoryOperationType::Retrieve => FrequencyBand::Gamma, // Pattern completion
            MemoryOperationType::Consolidate => FrequencyBand::Theta, // Memory replay
            MemoryOperationType::Idle => FrequencyBand::Alpha, // Resting state
        }
    }

    /// Get the optimal arousal level for this operation [0, 1]
    pub fn optimal_arousal(&self) -> f64 {
        match self {
            MemoryOperationType::Encode => 0.6, // Moderate arousal
            MemoryOperationType::Retrieve => 0.7, // Higher arousal for focus
            MemoryOperationType::Consolidate => 0.2, // Low arousal (sleep-like)
            MemoryOperationType::Idle => 0.3, // Relaxed
        }
    }
}
use super::types::{
    RoutingStrategy, RoutingPlan, RoutingOutcome,
    CombinedRoutingStrategy, PhaseLockedPlan, ScheduledOperation, ScheduledOpType,
    ConsciousnessAction, LatentConsciousnessState, OscillatoryState, OscillatoryPhase,
    ProcessingMode, Router, RouterStats,
};

// ═══════════════════════════════════════════════════════════════════════════
// PHASE WINDOW
// ═══════════════════════════════════════════════════════════════════════════

/// Defines an optimal execution window based on oscillatory phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseWindow {
    /// Target oscillatory phase
    pub target_phase: OscillatoryPhase,
    /// Processing mode required
    pub required_mode: ProcessingMode,
    /// Window duration (fraction of period)
    pub window_fraction: f64,
    /// Priority of this window
    pub priority: f64,
}

impl PhaseWindow {
    /// Check if current state is within this window
    pub fn is_active(&self, state: &OscillatoryState) -> bool {
        state.phase_category == self.target_phase
    }

    /// Get quality of match [0, 1]
    pub fn match_quality(&self, state: &OscillatoryState) -> f64 {
        if state.phase_category != self.target_phase {
            return 0.0;
        }

        let profile = state.phase_category.processing_profile();
        if profile.optimal_for.contains(&self.required_mode) {
            1.0
        } else {
            0.5
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for oscillatory router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryRouterConfig {
    /// Base frequency for consciousness oscillation (Hz)
    pub base_frequency: f64,
    /// Frequency adaptation rate
    pub frequency_adaptation: f64,
    /// Minimum amplitude for phase-locking
    pub min_amplitude: f64,
    /// Number of cycles to predict ahead
    pub prediction_cycles: usize,
    /// Enable phase-locked scheduling
    pub phase_locked_scheduling: bool,
    /// Weight for phase in routing decisions
    pub phase_weight: f64,
    /// Weight for magnitude in routing decisions
    pub magnitude_weight: f64,
}

impl Default for OscillatoryRouterConfig {
    fn default() -> Self {
        Self {
            base_frequency: 40.0, // Gamma band
            frequency_adaptation: 0.1,
            min_amplitude: 0.1,
            prediction_cycles: 3,
            phase_locked_scheduling: true,
            phase_weight: 0.4,
            magnitude_weight: 0.6,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for oscillatory routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OscillatoryRouterStats {
    /// Total routing decisions
    pub decisions_made: u64,
    /// Phase-locked decisions (hit optimal window)
    pub phase_locked_hits: u64,
    /// Phase misses (had to route at suboptimal phase)
    pub phase_misses: u64,
    /// Average phase coherence
    pub avg_coherence: f64,
    /// Average effective Φ
    pub avg_effective_phi: f64,
    /// Cycles completed
    pub cycles_completed: u64,
}

impl OscillatoryRouterStats {
    /// Get phase-locking accuracy
    pub fn phase_lock_accuracy(&self) -> f64 {
        if self.decisions_made == 0 {
            0.0
        } else {
            self.phase_locked_hits as f64 / self.decisions_made as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCHEDULED OPERATION (internal)
// ═══════════════════════════════════════════════════════════════════════════

/// Internal scheduled operation for phase-locked execution
#[derive(Debug, Clone)]
struct InternalScheduledOp {
    /// Unique identifier
    pub id: u64,
    /// Processing mode
    pub mode: ProcessingMode,
    /// Preferred execution window
    pub preferred_window: PhaseWindow,
    /// Maximum delay (cycles) before forced execution
    pub max_delay_cycles: usize,
    /// Current delay counter
    pub current_delay: usize,
    /// Priority
    pub priority: f64,
    /// Payload action
    pub payload: ConsciousnessAction,
}

impl InternalScheduledOp {
    /// Check if operation should execute now
    pub fn should_execute(&self, state: &OscillatoryState) -> bool {
        // Force execution if max delay reached
        if self.current_delay >= self.max_delay_cycles {
            return true;
        }

        // Execute if in optimal window
        self.preferred_window.is_active(state)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY ROUTER
// ═══════════════════════════════════════════════════════════════════════════

/// The Oscillatory Phase-Locked Router
///
/// Extends PredictiveRouter with phase awareness for true oscillatory routing.
/// Now includes multi-band oscillations (theta/alpha/beta/gamma) with cross-frequency coupling.
/// Phase 6C additions: Memory-oscillation gating, process resonance detection, attention spotlight
/// Phase 7 additions: Sleep-oscillation integration, delta/spindle bands, ultradian cycles
/// Phase 8 additions: Neuromodulatory dynamics (DA, 5-HT, NE, ACh), mood, motivation, learning
/// Phase 9 additions: Predictive consciousness (free energy, precision weighting, active inference)
pub struct OscillatoryRouter {
    /// Inner predictive router
    predictive_router: PredictiveRouter,
    /// Current oscillatory state (gamma-band primary)
    oscillatory_state: OscillatoryState,
    /// Multi-band oscillatory state (all frequency bands)
    multi_band_state: MultiBandState,
    /// Process resonance detector for phase-locked synchrony
    resonance_detector: ResonanceDetector,
    /// Attention spotlight for selective process enhancement
    attention_spotlight: AttentionSpotlight,
    /// Phase 7: Sleep-oscillation bridge for sleep/wake cycle integration
    sleep_bridge: SleepOscillationBridge,
    /// Phase 8: Neuromodulatory system (dopamine, serotonin, norepinephrine, acetylcholine)
    neuromodulator_system: NeuromodulatorSystem,
    /// Phase 9: Predictive consciousness (free energy principle integration)
    predictive_consciousness: PredictiveConsciousnessState,

    /// Phase 10: Global workspace (Baars' GWT - competition, ignition, broadcast)
    global_workspace: GlobalWorkspaceState,
    /// Configuration
    config: OscillatoryRouterConfig,
    /// Statistics
    stats: OscillatoryRouterStats,
    /// Scheduled operations queue
    scheduled_ops: VecDeque<InternalScheduledOp>,
    /// Next operation ID
    next_op_id: u64,
    /// Phase history for coherence calculation
    phase_history: VecDeque<f64>,
}

impl OscillatoryRouter {
    /// Create new oscillatory router
    pub fn new(config: OscillatoryRouterConfig) -> Self {
        // Initialize resonance detector with core cognitive processes
        let mut resonance_detector = ResonanceDetector::new();

        // Register default cognitive processes at gamma frequency
        resonance_detector.register_process("perception", 40.0);
        resonance_detector.register_process("attention", 40.0);
        resonance_detector.register_process("memory", 40.0);
        resonance_detector.register_process("reasoning", 40.0);
        resonance_detector.register_process("integration", 40.0);

        // Initialize attention spotlight with centered, moderate focus
        let attention_spotlight = AttentionSpotlight::new();

        Self {
            predictive_router: PredictiveRouter::new(PredictiveRouterConfig::default()),
            oscillatory_state: OscillatoryState::new(0.0, config.base_frequency, 0.3, 0.5),
            multi_band_state: MultiBandState::new(),
            resonance_detector,
            attention_spotlight,
            sleep_bridge: SleepOscillationBridge::new(), // Phase 7: Sleep-oscillation integration
            neuromodulator_system: NeuromodulatorSystem::new(), // Phase 8: Neuromodulatory dynamics
            predictive_consciousness: PredictiveConsciousnessState::new(), // Phase 9: Free energy principle
            global_workspace: GlobalWorkspaceState::new(), // Phase 10: Global workspace theory
            config,
            stats: OscillatoryRouterStats::default(),
            scheduled_ops: VecDeque::with_capacity(100),
            next_op_id: 0,
            phase_history: VecDeque::with_capacity(100),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 7: SLEEP-OSCILLATION INTEGRATION METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Transition to a new sleep stage
    /// This modulates all oscillatory bands according to sleep architecture
    pub fn set_sleep_stage(&mut self, stage: SleepStage) {
        self.sleep_bridge.transition_to(stage);
    }

    /// Get current sleep stage
    pub fn sleep_stage(&self) -> SleepStage {
        self.sleep_bridge.current_stage
    }

    /// Check if the system is currently in a sleep state
    pub fn is_sleeping(&self) -> bool {
        self.sleep_bridge.current_stage.is_sleeping()
    }

    /// Get sleep pressure (homeostatic sleep drive) [0, 1]
    pub fn sleep_pressure(&self) -> f64 {
        self.sleep_bridge.sleep_pressure
    }

    /// Get memory consolidation efficiency for current state [0, 1]
    pub fn consolidation_efficiency(&self) -> f64 {
        self.sleep_bridge.consolidation_efficiency()
    }

    /// Check if currently in optimal memory transfer window
    pub fn in_memory_transfer_window(&self) -> bool {
        self.sleep_bridge.in_memory_transfer_window()
    }

    /// Get consciousness level based on sleep stage [0, 1]
    /// Based on IIT: Φ decreases during deep sleep
    pub fn consciousness_level(&self) -> f64 {
        self.sleep_bridge.current_stage.consciousness_level()
    }

    /// Get sleep-oscillation summary for MindState integration
    pub fn sleep_summary(&self) -> SleepOscillationSummary {
        self.sleep_bridge.summary()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 8: NEUROMODULATORY DYNAMICS METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current dopamine level (effective, includes receptor sensitivity)
    pub fn dopamine(&self) -> f64 {
        self.neuromodulator_system.dopamine.effective_level()
    }

    /// Get current serotonin level
    pub fn serotonin(&self) -> f64 {
        self.neuromodulator_system.serotonin.effective_level()
    }

    /// Get current norepinephrine level
    pub fn norepinephrine(&self) -> f64 {
        self.neuromodulator_system.norepinephrine.effective_level()
    }

    /// Get current acetylcholine level
    pub fn acetylcholine(&self) -> f64 {
        self.neuromodulator_system.acetylcholine.effective_level()
    }

    /// Get current mood valence (-1 = negative, +1 = positive)
    pub fn mood(&self) -> f64 {
        self.neuromodulator_system.mood_valence
    }

    /// Get current motivation drive [0, 1]
    pub fn motivation(&self) -> f64 {
        self.neuromodulator_system.motivation_drive
    }

    /// Get current stress level [0, 1]
    pub fn stress(&self) -> f64 {
        self.neuromodulator_system.stress_level
    }

    /// Get learning rate modifier [0.5, 2.0]
    pub fn learning_rate_modifier(&self) -> f64 {
        self.neuromodulator_system.learning_rate_mod
    }

    /// Process a reward signal (updates dopamine and serotonin)
    /// reward: absolute reward value [0, 1]
    /// prediction_error: reward - expected_reward [-1, 1]
    pub fn process_reward(&mut self, reward: f64, prediction_error: f64) {
        self.neuromodulator_system.process_reward(reward, prediction_error);
    }

    /// Process a stress/threat signal (updates norepinephrine and serotonin)
    pub fn process_stress(&mut self, intensity: f64) {
        self.neuromodulator_system.process_stress(intensity);
    }

    /// Process a novelty/curiosity signal (updates dopamine, NE, ACh)
    pub fn process_novelty(&mut self, novelty: f64) {
        self.neuromodulator_system.process_novelty(novelty);
    }

    /// Release a specific neuromodulator manually
    pub fn release_neuromodulator(&mut self, nm: Neuromodulator, intensity: f64) {
        self.neuromodulator_system.release(nm, intensity);
    }

    /// Get neuromodulator summary for MindState integration
    pub fn neuromodulator_summary(&self) -> NeuromodulatorSummary {
        self.neuromodulator_system.summary()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 9: PREDICTIVE CONSCIOUSNESS METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get total free energy (variational free energy across all levels)
    pub fn free_energy(&self) -> f64 {
        self.predictive_consciousness.total_free_energy
    }

    /// Get weighted prediction error
    pub fn prediction_error(&self) -> f64 {
        self.predictive_consciousness.weighted_prediction_error
    }

    /// Get surprise (negative log probability of current observation)
    pub fn surprise(&self) -> f64 {
        self.predictive_consciousness.surprise
    }

    /// Get global precision gain (influenced by norepinephrine/arousal)
    pub fn precision_gain(&self) -> f64 {
        self.predictive_consciousness.precision_gain
    }

    /// Get expected free energy for current strategy
    pub fn expected_free_energy(&self) -> f64 {
        self.predictive_consciousness.expected_free_energy
    }

    /// Get epistemic value (information gain potential)
    pub fn epistemic_value(&self) -> f64 {
        self.predictive_consciousness.epistemic_value
    }

    /// Get pragmatic value (goal-alignment measure)
    pub fn pragmatic_value(&self) -> f64 {
        self.predictive_consciousness.pragmatic_value
    }

    /// Get exploration drive (0=exploit, 1=explore)
    pub fn exploration_drive(&self) -> f64 {
        self.predictive_consciousness.exploration_drive
    }

    /// Get precision at a specific hierarchical level
    pub fn level_precision(&self, level: &str) -> f64 {
        match level {
            "sensory" => self.predictive_consciousness.sensory.precision,
            "perceptual" => self.predictive_consciousness.perceptual.precision,
            "conceptual" => self.predictive_consciousness.conceptual.precision,
            "metacognitive" => self.predictive_consciousness.metacognitive.precision,
            _ => 0.5,
        }
    }

    /// Get prediction at a specific hierarchical level
    pub fn level_prediction(&self, level: &str) -> f64 {
        match level {
            "sensory" => self.predictive_consciousness.sensory.prediction,
            "perceptual" => self.predictive_consciousness.perceptual.prediction,
            "conceptual" => self.predictive_consciousness.conceptual.prediction,
            "metacognitive" => self.predictive_consciousness.metacognitive.prediction,
            _ => 0.5,
        }
    }

    /// Process goal and uncertainty for active inference
    /// goal_alignment: how well current state matches goals [0, 1]
    /// uncertainty: current uncertainty in beliefs [0, 1]
    pub fn process_active_inference(&mut self, goal_alignment: f64, uncertainty: f64) {
        self.predictive_consciousness.compute_expected_free_energy(goal_alignment, uncertainty);
    }

    /// Check if the system is in high-surprise state (unexpected observations)
    pub fn is_surprised(&self) -> bool {
        self.predictive_consciousness.surprise > 0.5
    }

    /// Check if the system should explore (high uncertainty, low goal alignment)
    pub fn should_explore(&self) -> bool {
        self.predictive_consciousness.exploration_drive > 0.6
    }

    /// Get predictive consciousness summary for MindState integration
    pub fn predictive_summary(&self) -> PredictiveConsciousnessSummary {
        self.predictive_consciousness.summary()
    }

    // =========================================================================
    // GLOBAL WORKSPACE ACCESSORS (Phase 10)
    // =========================================================================

    /// Check if workspace is currently ignited (conscious access occurring)
    pub fn is_ignited(&self) -> bool {
        self.global_workspace.is_ignited
    }

    /// Get currently broadcasting processor (if ignited)
    pub fn broadcasting_processor(&self) -> Option<WorkspaceProcessor> {
        self.global_workspace.broadcasting_processor
    }

    /// Get current broadcast strength [0, 1]
    pub fn broadcast_strength(&self) -> f64 {
        self.global_workspace.broadcast_strength
    }

    /// Get current ignition threshold
    pub fn ignition_threshold(&self) -> f64 {
        self.global_workspace.ignition_threshold
    }

    /// Get workspace integration level [0, 1]
    pub fn workspace_integration(&self) -> f64 {
        self.global_workspace.integration
    }

    /// Get number of active processor coalitions
    pub fn coalition_count(&self) -> usize {
        self.global_workspace.active_coalitions.len()
    }

    /// Get activation level for a specific processor
    pub fn processor_activation(&self, processor: WorkspaceProcessor) -> f64 {
        self.global_workspace.processor_activation(processor)
    }

    /// Get total number of ignition events
    pub fn ignition_count(&self) -> u64 {
        self.global_workspace.ignition_count
    }

    /// Get total broadcast time in seconds
    pub fn total_broadcast_time(&self) -> f64 {
        self.global_workspace.total_broadcast_time
    }

    /// Get global workspace summary for MindState integration
    pub fn workspace_summary(&self) -> GlobalWorkspaceSummary {
        self.global_workspace.summary()
    }

    /// Update oscillatory state from observations
    pub fn observe_state(&mut self, phi: f64, dt: f64) {
        // Update Φ
        self.oscillatory_state.phi = phi;

        // ===== PHI-OSCILLATION COUPLING =====
        // Higher Φ strengthens gamma synchronization (neuroscientifically validated)
        // - High Φ (>0.5): Increase frequency toward 45Hz (hyper-synchronized)
        // - Low Φ (<0.3): Decrease frequency toward 35Hz (desynchronized)
        // - Optimal Φ (~0.4-0.5): Maintain base 40Hz gamma
        let target_frequency = if phi > 0.5 {
            // Strong integration: boost toward 45Hz
            self.config.base_frequency + (phi - 0.5) * 10.0  // Up to 45Hz at Φ=1.0
        } else if phi < 0.3 {
            // Weak integration: drop toward 35Hz
            self.config.base_frequency - (0.3 - phi) * 16.67  // Down to 35Hz at Φ=0.0
        } else {
            // Optimal range: maintain base frequency
            self.config.base_frequency
        };

        // Smooth frequency transition using adaptation rate
        self.oscillatory_state.frequency = self.oscillatory_state.frequency
            + self.config.frequency_adaptation * (target_frequency - self.oscillatory_state.frequency);

        // Advance phase
        self.oscillatory_state.advance(dt);

        // Record phase for coherence
        self.phase_history.push_back(self.oscillatory_state.phase);
        while self.phase_history.len() > 100 {
            self.phase_history.pop_front();
        }

        // Update coherence estimate (modulated by Φ)
        // Higher Φ naturally leads to better phase coherence
        let raw_coherence = self.estimate_coherence();
        let phi_coherence_boost = (phi - 0.3).max(0.0) * 0.2;  // Up to 20% boost from high Φ
        self.oscillatory_state.coherence = (raw_coherence + phi_coherence_boost).min(1.0);

        // Update stats
        let n = self.stats.cycles_completed as f64 + 1.0;
        self.stats.avg_effective_phi =
            (self.stats.avg_effective_phi * (n - 1.0) + self.oscillatory_state.effective_phi()) / n;
        self.stats.avg_coherence =
            (self.stats.avg_coherence * (n - 1.0) + self.oscillatory_state.coherence) / n;

        // ===== MULTI-BAND UPDATE =====
        // Advance all frequency bands with cross-frequency coupling
        self.multi_band_state.advance(dt);

        // Sync gamma band state with primary oscillatory state
        if let Some(gamma) = self.multi_band_state.bands.get_mut(&FrequencyBand::Gamma) {
            gamma.frequency = self.oscillatory_state.frequency;
            gamma.coherence = self.oscillatory_state.coherence;
            gamma.phase = self.oscillatory_state.phase;
        }

        // Modulate theta-gamma coupling based on Φ
        // Higher Φ = stronger theta-gamma coupling (better memory binding)
        self.multi_band_state.theta_gamma_coupling = 0.3 + phi * 0.5; // 0.3 to 0.8

        // ===== PROCESS RESONANCE UPDATE (Phase 6C) =====
        // Advance resonance detector and update process synchrony
        self.resonance_detector.advance(dt);

        // ===== ATTENTION SPOTLIGHT UPDATE (Phase 6C) =====
        // Advance attention spotlight sampling and focus decay
        self.attention_spotlight.advance(dt);

        // ===== SLEEP-OSCILLATION UPDATE (Phase 7) =====
        // Update sleep bridge and modulate bands based on sleep stage
        self.sleep_bridge.update(dt, &mut self.multi_band_state);

        // Modulate Φ based on sleep stage (consciousness reduction during deep sleep)
        // This is bidirectional: sleep stage affects effective Φ perception
        let consciousness_mod = self.sleep_bridge.current_stage.consciousness_level();
        self.oscillatory_state.phi *= consciousness_mod;

        // ===== NEUROMODULATORY DYNAMICS UPDATE (Phase 8) =====
        // Update neuromodulator kinetics (reuptake, receptor adaptation)
        self.neuromodulator_system.update(dt);

        // Adapt neuromodulators to current sleep stage
        self.neuromodulator_system.adapt_to_sleep_stage(self.sleep_bridge.current_stage);

        // Apply neuromodulator effects on oscillatory bands
        // Each neuromodulator enhances or suppresses specific frequency bands
        for band in [FrequencyBand::Theta, FrequencyBand::Alpha, FrequencyBand::Beta, FrequencyBand::Gamma, FrequencyBand::Delta] {
            let modulation = self.neuromodulator_system.band_modulation(band);
            if let Some(band_state) = self.multi_band_state.bands.get_mut(&band) {
                // Modulate power based on neuromodulator balance
                band_state.power = (band_state.power * modulation).clamp(0.0, 1.0);
            }
        }

        // Norepinephrine directly affects global arousal
        let ne_arousal = self.neuromodulator_system.norepinephrine.effective_level();
        let current_arousal = self.multi_band_state.arousal;
        // Blend current arousal with NE-driven arousal (NE has 40% influence)
        let blended_arousal = current_arousal * 0.6 + ne_arousal * 0.4;
        self.multi_band_state.arousal = blended_arousal.clamp(0.0, 1.0);

        // Acetylcholine boosts attention and learning-related gamma
        let ach_level = self.neuromodulator_system.acetylcholine.effective_level();
        if ach_level > 0.5 {
            // High ACh: boost gamma for enhanced attention/learning
            if let Some(gamma) = self.multi_band_state.bands.get_mut(&FrequencyBand::Gamma) {
                gamma.power = (gamma.power * (1.0 + (ach_level - 0.5) * 0.4)).min(1.0);
            }
        }

        // Dopamine influences motivation-related processing
        // High DA: more willing to engage in effortful processing
        let da_level = self.neuromodulator_system.dopamine.effective_level();
        if da_level > 0.4 {
            // Boost beta (action/motor) when motivated
            if let Some(beta) = self.multi_band_state.bands.get_mut(&FrequencyBand::Beta) {
                beta.power = (beta.power * (1.0 + (da_level - 0.4) * 0.3)).min(1.0);
            }
        }

        // ===== PREDICTIVE CONSCIOUSNESS UPDATE (Phase 9) =====
        // Process observation through hierarchical predictive model
        // Uses: phi (integration), coherence, and meta-awareness
        let coherence = self.oscillatory_state.coherence;
        let meta_awareness = self.attention_spotlight.primary_intensity(); // Use attention as proxy for meta-awareness

        self.predictive_consciousness.observe(phi, coherence, meta_awareness, dt);

        // Apply neuromodulator effects to precision weighting
        // DA: increases precision on salient/rewarding stimuli
        // NE: globally increases precision (arousal)
        // ACh: increases precision for learning
        let da = self.neuromodulator_system.dopamine.effective_level();
        let serotonin = self.neuromodulator_system.serotonin.effective_level();
        let ne = self.neuromodulator_system.norepinephrine.effective_level();
        let ach = self.neuromodulator_system.acetylcholine.effective_level();

        self.predictive_consciousness.apply_neuromodulators(da, serotonin, ne, ach);

        // Compute expected free energy for active inference
        // Goal alignment: based on current phi vs target (0.7 is optimal consciousness)
        let goal_alignment = 1.0 - (phi - 0.7).abs();
        // Uncertainty: based on error variance in predictions
        let uncertainty = self.predictive_consciousness.error_variance.sqrt().min(1.0);

        self.predictive_consciousness.compute_expected_free_energy(goal_alignment, uncertainty);

        // Free energy influences oscillatory dynamics
        // High surprise (high FE) → boost arousal/attention
        if self.predictive_consciousness.surprise > 0.3 {
            // Unexpected observation: boost NE for arousal
            let surprise_boost = (self.predictive_consciousness.surprise - 0.3) * 0.5;
            self.multi_band_state.arousal = (self.multi_band_state.arousal + surprise_boost).min(1.0);
        }

        // Exploration drive influences gamma/beta balance
        // High exploration → more gamma (information seeking)
        // High exploitation → more beta (action maintenance)
        if self.predictive_consciousness.exploration_drive > 0.6 {
            if let Some(gamma) = self.multi_band_state.bands.get_mut(&FrequencyBand::Gamma) {
                gamma.power = (gamma.power * 1.1).min(1.0);
            }
        } else if self.predictive_consciousness.exploration_drive < 0.4 {
            if let Some(beta) = self.multi_band_state.bands.get_mut(&FrequencyBand::Beta) {
                beta.power = (beta.power * 1.1).min(1.0);
            }
        }

        // ===== GLOBAL WORKSPACE UPDATE (Phase 10) =====
        // Bernard Baars' Global Workspace Theory: competition, ignition, broadcast
        // Gamma synchronization IS the broadcast mechanism

        // Get oscillatory state for workspace update
        let gamma_power = self.multi_band_state.bands.get(&FrequencyBand::Gamma)
            .map(|b| b.power)
            .unwrap_or(0.5);
        let gamma_coherence = self.multi_band_state.bands.get(&FrequencyBand::Gamma)
            .map(|b| b.coherence)
            .unwrap_or(0.5);
        let theta_gamma_coupling = self.multi_band_state.theta_gamma_coupling;
        let attention_intensity = self.attention_spotlight.primary_intensity();
        let surprise = self.predictive_consciousness.surprise;

        // Update processor activations based on current consciousness state
        self.global_workspace.update_processors(
            phi,
            attention_intensity,
            gamma_power,
            theta_gamma_coupling,
            da,      // Already computed above
            ne,      // Already computed above
            surprise,
            dt,
        );

        // Update coalitions based on gamma coherence
        // High gamma coherence enables coalition formation
        self.global_workspace.update_coalitions(gamma_coherence);

        // Check for ignition and handle broadcast dynamics
        // Neuromodulators affect ignition threshold
        self.global_workspace.update_ignition(
            gamma_power,
            gamma_coherence,
            da,
            serotonin,
            dt,
        );

        // Bidirectional influence: Global workspace affects oscillatory dynamics
        if self.global_workspace.is_ignited {
            // During ignition, boost gamma power and coherence (broadcast mechanism)
            if let Some(gamma) = self.multi_band_state.bands.get_mut(&FrequencyBand::Gamma) {
                gamma.power = (gamma.power * 1.15).min(1.0);
                gamma.coherence = (gamma.coherence * 1.1).min(1.0);
            }

            // Suppress alpha (less idling during conscious access)
            if let Some(alpha) = self.multi_band_state.bands.get_mut(&FrequencyBand::Alpha) {
                alpha.power *= 0.9;
            }

            // Ignition boosts attention to the broadcasting content
            self.attention_spotlight.shift_attention(
                self.global_workspace.broadcast_strength,
                attention_intensity + 0.1, // Slightly boost intensity
            );
        }

        // Prediction error can trigger ignition (unexpected = salient)
        if self.predictive_consciousness.surprise > 0.5 && !self.global_workspace.is_ignited {
            // High surprise content competes strongly for workspace
            // This is handled in update_processors via the surprise parameter
        }
    }

    /// Set arousal level (shifts dominant frequency band)
    ///
    /// - Low arousal (0.0-0.3): Alpha/Theta dominant (relaxed, drowsy)
    /// - Medium arousal (0.3-0.6): Beta dominant (alert, focused)
    /// - High arousal (0.6-1.0): Gamma dominant (intense focus, consciousness)
    pub fn set_arousal(&mut self, arousal: f64) {
        self.multi_band_state.set_arousal(arousal);
    }

    /// Set attention level (boosts gamma, suppresses alpha)
    ///
    /// High attention gates gamma oscillations for focused processing
    /// while suppressing alpha (idling) rhythms
    pub fn set_attention(&mut self, attention: f64) {
        self.multi_band_state.set_attention(attention);
    }

    /// Get the multi-band oscillatory state
    pub fn multi_band_state(&self) -> &MultiBandState {
        &self.multi_band_state
    }

    /// Get multi-band summary for monitoring
    pub fn multi_band_summary(&self) -> MultiBandSummary {
        self.multi_band_state.summary()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PROCESS RESONANCE DETECTION (Phase 6C)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get the resonance detector
    pub fn resonance_detector(&self) -> &ResonanceDetector {
        &self.resonance_detector
    }

    /// Get mutable resonance detector for registering new processes
    pub fn resonance_detector_mut(&mut self) -> &mut ResonanceDetector {
        &mut self.resonance_detector
    }

    /// Register a new cognitive process for resonance tracking
    pub fn register_process(&mut self, name: &str, frequency: f64) -> ProcessId {
        self.resonance_detector.register_process(name, frequency)
    }

    /// Update a process's oscillatory state
    pub fn update_process(&mut self, id: ProcessId, phase: f64, amplitude: f64, active: bool) {
        self.resonance_detector.update_process(id, phase, amplitude, active);
    }

    /// Get overall system resonance (mean PLV)
    pub fn system_resonance(&self) -> f64 {
        self.resonance_detector.system_resonance()
    }

    /// Get resonance-based integration boost factor
    /// High resonance → enhanced binding → higher effective Φ
    pub fn resonance_integration_boost(&self) -> f64 {
        self.resonance_detector.integration_boost()
    }

    /// Get resonance summary for monitoring
    pub fn resonance_summary(&self) -> ResonanceDetectorSummary {
        self.resonance_detector.summary()
    }

    /// Get the primary resonant cluster (largest synchronized group)
    pub fn primary_cluster(&self) -> Option<&ResonantCluster> {
        self.resonance_detector.get_primary_cluster()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ATTENTION SPOTLIGHT SYSTEM (Phase 6C)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get the attention spotlight
    pub fn attention_spotlight(&self) -> &AttentionSpotlight {
        &self.attention_spotlight
    }

    /// Get mutable attention spotlight
    pub fn attention_spotlight_mut(&mut self) -> &mut AttentionSpotlight {
        &mut self.attention_spotlight
    }

    /// Focus attention on a position in attention space [0, 1]
    pub fn focus_attention(&mut self, position: f64, intensity: f64) {
        self.attention_spotlight.focus_on(position, intensity);
    }

    /// Focus attention on a specific process
    pub fn focus_on_process(&mut self, process_id: ProcessId, intensity: f64) {
        self.attention_spotlight.focus_on_process(process_id, intensity);
    }

    /// Add a secondary attention focus (for split attention)
    pub fn add_attention_focus(&mut self, position: f64, intensity: f64) -> bool {
        let focus = AttentionFocus::new(position, intensity);
        self.attention_spotlight.add_secondary_focus(focus)
    }

    /// Consolidate all attention to primary focus
    pub fn consolidate_attention(&mut self) {
        self.attention_spotlight.consolidate_attention();
    }

    /// Get enhancement factor for a process at given position
    /// Returns [0, 1] where 0.5 = neutral, >0.5 = enhanced, <0.5 = suppressed
    pub fn get_attention_enhancement(&self, process_id: ProcessId, position: f64) -> f64 {
        self.attention_spotlight.get_enhancement(process_id, position)
    }

    /// Check if attention is in uptake phase (good for encoding)
    pub fn in_attention_uptake(&self) -> bool {
        self.attention_spotlight.in_uptake_phase()
    }

    /// Get attention spotlight summary
    pub fn attention_summary(&self) -> AttentionSpotlightSummary {
        self.attention_spotlight.summary()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MEMORY-OSCILLATION INTEGRATION (Phase 6C)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Check if current oscillatory state is optimal for memory encoding
    ///
    /// Memory encoding is optimal during theta-gamma coupling:
    /// - Theta at peak phase (episodic binding context)
    /// - Gamma power high (detail encoding)
    /// - Theta-gamma coupling strong
    ///
    /// Returns: Encoding quality factor [0, 1] where 1 = perfect encoding window
    pub fn memory_encoding_quality(&self) -> f64 {
        let mb = &self.multi_band_state;

        // Check theta-gamma coupling window
        if !mb.in_memory_encoding_window() {
            return 0.2; // Suboptimal but still possible
        }

        // Calculate encoding quality based on:
        // 1. Theta-gamma coupling strength (40%)
        // 2. Gamma power (30%)
        // 3. Overall coherence (30%)
        let coupling_factor = mb.theta_gamma_coupling;
        let gamma_factor = mb.bands.get(&FrequencyBand::Gamma)
            .map(|g| g.power).unwrap_or(0.5);
        let coherence_factor = mb.overall_coherence();

        0.4 * coupling_factor + 0.3 * gamma_factor + 0.3 * coherence_factor
    }

    /// Check if current oscillatory state is optimal for memory retrieval
    ///
    /// Memory retrieval is optimal during high gamma states:
    /// - Gamma power high (pattern completion)
    /// - Gamma coherence high (stable recall)
    /// - Attention high (focused retrieval)
    ///
    /// Returns: Retrieval quality factor [0, 1] where 1 = perfect retrieval window
    pub fn memory_retrieval_quality(&self) -> f64 {
        let mb = &self.multi_band_state;

        // Get gamma state
        let gamma = mb.bands.get(&FrequencyBand::Gamma);
        let gamma_power = gamma.map(|g| g.power).unwrap_or(0.5);
        let gamma_coherence = gamma.map(|g| g.coherence).unwrap_or(0.5);

        // Retrieval quality based on:
        // 1. Gamma power (40%) - pattern completion strength
        // 2. Gamma coherence (30%) - stable recall
        // 3. Attention level (30%) - focused retrieval
        0.4 * gamma_power + 0.3 * gamma_coherence + 0.3 * mb.attention
    }

    /// Check if current oscillatory state is optimal for memory consolidation
    ///
    /// Memory consolidation is optimal during low-arousal states:
    /// - Low gamma (reduced active processing)
    /// - High theta (memory replay)
    /// - Low arousal (sleep-like state)
    ///
    /// Returns: Consolidation quality factor [0, 1] where 1 = perfect consolidation window
    pub fn memory_consolidation_quality(&self) -> f64 {
        let mb = &self.multi_band_state;

        // Get band states
        let gamma_power = mb.bands.get(&FrequencyBand::Gamma)
            .map(|g| g.power).unwrap_or(0.5);
        let theta_power = mb.bands.get(&FrequencyBand::Theta)
            .map(|t| t.power).unwrap_or(0.5);

        // Consolidation quality based on:
        // 1. Low gamma power (40%) - reduced interference
        // 2. High theta power (35%) - memory replay
        // 3. Low arousal (25%) - offline processing
        let gamma_factor = 1.0 - gamma_power; // Inverse: low gamma is good
        let arousal_factor = 1.0 - mb.arousal; // Inverse: low arousal is good

        0.4 * gamma_factor + 0.35 * theta_power + 0.25 * arousal_factor
    }

    /// Get recommended memory operation based on current oscillatory state
    ///
    /// Returns the operation type that would be most effective right now
    pub fn recommended_memory_operation(&self) -> MemoryOperationType {
        let encoding = self.memory_encoding_quality();
        let retrieval = self.memory_retrieval_quality();
        let consolidation = self.memory_consolidation_quality();

        // Find the best operation
        if encoding >= retrieval && encoding >= consolidation && encoding > 0.5 {
            MemoryOperationType::Encode
        } else if retrieval >= consolidation && retrieval > 0.5 {
            MemoryOperationType::Retrieve
        } else if consolidation > 0.5 {
            MemoryOperationType::Consolidate
        } else {
            MemoryOperationType::Idle
        }
    }

    /// Get memory gating signal for operation scheduling
    ///
    /// This signal can be used to gate memory operations:
    /// - High signal = proceed with operation
    /// - Low signal = defer operation
    ///
    /// # Arguments
    /// * `operation` - The type of memory operation to gate
    pub fn memory_gate_signal(&self, operation: MemoryOperationType) -> f64 {
        match operation {
            MemoryOperationType::Encode => self.memory_encoding_quality(),
            MemoryOperationType::Retrieve => self.memory_retrieval_quality(),
            MemoryOperationType::Consolidate => self.memory_consolidation_quality(),
            MemoryOperationType::Idle => 1.0, // Always allow idle
        }
    }

    /// Apply hormone modulation to oscillatory dynamics
    ///
    /// Hormones affect gamma synchronization:
    /// - **Cortisol (stress)**: Disrupts gamma rhythm, reduces coherence and frequency stability
    /// - **Dopamine (reward)**: Enhances gamma synchronization, increases coherence
    /// - **Serotonin**: Stabilizes rhythm, reduces frequency variance
    /// - **Norepinephrine (arousal)**: Increases amplitude/intensity
    ///
    /// # Arguments
    /// * `cortisol` - Stress hormone level [0, 1]
    /// * `dopamine` - Reward hormone level [0, 1]
    pub fn modulate_hormones(&mut self, cortisol: f32, dopamine: f32) {
        // ===== HORMONE-OSCILLATION COUPLING =====
        // Based on neuroscience research on stress/reward effects on gamma oscillations

        // Cortisol DISRUPTS gamma synchronization
        // - High cortisol → adds noise to phase, reduces coherence
        // - Simulates stress-induced cognitive fragmentation
        let cortisol_disruption = cortisol as f64 * 0.15;  // Up to 15% coherence reduction
        self.oscillatory_state.coherence =
            (self.oscillatory_state.coherence - cortisol_disruption).max(0.2);

        // Dopamine ENHANCES gamma synchronization
        // - High dopamine → tightens phase-locking, boosts coherence
        // - Simulates reward-enhanced focus and attention
        let dopamine_enhancement = dopamine as f64 * 0.1;  // Up to 10% coherence boost
        self.oscillatory_state.coherence =
            (self.oscillatory_state.coherence + dopamine_enhancement).min(1.0);

        // Combined effect on frequency stability
        // High stress → frequency jitter; High dopamine → stable frequency
        let stability_factor = (1.0 - cortisol as f64 * 0.5) * (1.0 + dopamine as f64 * 0.3);

        // Apply stability to frequency (closer to base when stable, more variance when stressed)
        let frequency_deviation = self.oscillatory_state.frequency - self.config.base_frequency;
        self.oscillatory_state.frequency = self.config.base_frequency
            + frequency_deviation * stability_factor;
    }

    /// Estimate phase coherence from history
    fn estimate_coherence(&self) -> f64 {
        if self.phase_history.len() < 10 {
            return 1.0;
        }

        // Calculate coherence from phase differences
        let expected_diff = 2.0 * PI * self.config.base_frequency * 0.01; // assuming 10ms samples
        let mut diff_coherence = 0.0;

        for i in 1..self.phase_history.len() {
            let diff = self.phase_history[i] - self.phase_history[i - 1];
            let normalized_diff = ((diff + PI) % (2.0 * PI)) - PI;
            diff_coherence += (1.0 - (normalized_diff - expected_diff).abs() / PI).max(0.0);
        }

        diff_coherence / (self.phase_history.len() - 1) as f64
    }

    /// Create phase-locked routing plan
    pub fn plan_phase_locked(&mut self, current_state: &LatentConsciousnessState) -> PhaseLockedPlan {
        // Get magnitude-based plan from inner router
        let magnitude_plan = self.predictive_router.plan_route(current_state);

        // Update oscillatory state
        self.oscillatory_state.phi = current_state.phi;

        // Predict oscillatory states for upcoming cycles
        let period = 1.0 / self.config.base_frequency;
        let mut predicted_states = Vec::new();

        for i in 0..self.config.prediction_cycles {
            let dt = period * (i as f64 + 0.25); // Sample at quarter periods
            predicted_states.push(self.oscillatory_state.predict(dt));
        }

        // Find optimal execution windows
        let mut execution_windows = Vec::new();
        for phase in [OscillatoryPhase::Peak, OscillatoryPhase::Rising,
                      OscillatoryPhase::Falling, OscillatoryPhase::Trough] {
            let time_until = phase.time_until(self.oscillatory_state.phase, self.config.base_frequency);
            execution_windows.push((phase, time_until));
        }

        // Sort by time
        execution_windows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Determine combined strategy
        let phase_mode = self.oscillatory_state.phase_category.processing_profile()
            .optimal_for.first().copied().unwrap_or(ProcessingMode::Integration);

        let magnitude_factor = magnitude_plan.current_strategy.resource_factor();
        let phase_factor = self.oscillatory_state.phase_category.processing_profile().integration_capacity;

        let combined_factor = self.config.magnitude_weight * magnitude_factor
            + self.config.phase_weight * phase_factor;

        let combined_confidence = self.oscillatory_state.coherence *
            magnitude_plan.predictions.first().map(|p| p.confidence).unwrap_or(0.5);

        let combined_strategy = CombinedRoutingStrategy {
            magnitude_strategy: magnitude_plan.current_strategy,
            phase_mode,
            resource_factor: combined_factor,
            confidence: combined_confidence,
        };

        // Calculate expected quality
        let expected_quality = combined_confidence * combined_factor;

        PhaseLockedPlan {
            magnitude_plan,
            current_phase: self.oscillatory_state.phase,
            predicted_states,
            execution_windows,
            combined_strategy,
            phase_coherence: self.oscillatory_state.coherence,
        }
    }

    /// Schedule operation for phase-locked execution
    pub fn schedule_operation(
        &mut self,
        mode: ProcessingMode,
        action: ConsciousnessAction,
        priority: f64,
    ) -> u64 {
        let target_phase = match mode {
            ProcessingMode::Binding | ProcessingMode::Integration => OscillatoryPhase::Peak,
            ProcessingMode::InputGathering | ProcessingMode::Attention => OscillatoryPhase::Rising,
            ProcessingMode::OutputGeneration | ProcessingMode::Consolidation => OscillatoryPhase::Falling,
            ProcessingMode::Reset | ProcessingMode::Maintenance => OscillatoryPhase::Trough,
            // Additional modes mapped to appropriate phases
            ProcessingMode::Serial | ProcessingMode::Parallel => OscillatoryPhase::Peak,
            ProcessingMode::Oscillatory => OscillatoryPhase::Rising,
        };

        let op = InternalScheduledOp {
            id: self.next_op_id,
            mode,
            preferred_window: PhaseWindow {
                target_phase,
                required_mode: mode,
                window_fraction: 0.25,
                priority,
            },
            max_delay_cycles: 3,
            current_delay: 0,
            priority,
            payload: action,
        };

        self.next_op_id += 1;
        let id = op.id;
        self.scheduled_ops.push_back(op);

        id
    }

    /// Execute ready operations
    pub fn execute_ready(&mut self) -> Vec<(u64, ConsciousnessAction)> {
        let mut executed = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(mut op) = self.scheduled_ops.pop_front() {
            if op.should_execute(&self.oscillatory_state) {
                self.stats.decisions_made += 1;

                if op.preferred_window.is_active(&self.oscillatory_state) {
                    self.stats.phase_locked_hits += 1;
                } else {
                    self.stats.phase_misses += 1;
                }

                executed.push((op.id, op.payload));
            } else {
                op.current_delay += 1;
                remaining.push_back(op);
            }
        }

        self.scheduled_ops = remaining;
        executed
    }

    /// Run one cycle of the oscillatory router
    pub fn cycle(&mut self, dt: f64) {
        // Advance oscillatory state
        self.oscillatory_state.advance(dt);

        // Check for cycle completion
        if self.oscillatory_state.phase_category == OscillatoryPhase::Peak
            && self.oscillatory_state.phase.abs() < 0.1 {
            self.stats.cycles_completed += 1;
        }

        // Run inner router cycle
        self.predictive_router.cycle();
    }

    /// Get current oscillatory state
    pub fn oscillatory_state(&self) -> &OscillatoryState {
        &self.oscillatory_state
    }

    /// Get statistics
    pub fn get_stats(&self) -> &OscillatoryRouterStats {
        &self.stats
    }

    /// Get summary
    pub fn summary(&self) -> OscillatoryRouterSummary {
        let mb = self.multi_band_state.summary();
        let rs = self.resonance_detector.summary();
        let as_ = self.attention_spotlight.summary();
        let sleep = self.sleep_bridge.summary();
        let nm_summary = self.neuromodulator_system.summary(); // Phase 8

        // Get delta and spindle band powers
        let delta_power = self.multi_band_state.bands
            .get(&FrequencyBand::Delta)
            .map(|s| s.power)
            .unwrap_or(0.0);
        let spindle_power = self.multi_band_state.bands
            .get(&FrequencyBand::SleepSpindle)
            .map(|s| s.power)
            .unwrap_or(0.0);

        OscillatoryRouterSummary {
            current_phase: self.oscillatory_state.phase_category,
            current_phi: self.oscillatory_state.phi,
            effective_phi: self.oscillatory_state.effective_phi(),
            coherence: self.oscillatory_state.coherence,
            frequency: self.oscillatory_state.frequency,
            phase_lock_accuracy: self.stats.phase_lock_accuracy(),
            cycles_completed: self.stats.cycles_completed,
            pending_operations: self.scheduled_ops.len(),
            predictive_summary: self.predictive_router.summary(),
            // Multi-band fields
            dominant_band: mb.dominant_band,
            gamma_power: mb.gamma_power,
            theta_power: mb.theta_power,
            alpha_power: mb.alpha_power,
            theta_gamma_coupling: mb.theta_gamma_coupling,
            arousal: mb.arousal,
            attention: mb.attention,
            in_memory_window: mb.in_memory_window,
            // Resonance detection fields (Phase 6C)
            system_resonance: rs.system_resonance,
            resonance_integration_boost: rs.integration_boost,
            resonating_pairs: rs.resonating_pairs,
            cluster_count: rs.cluster_count,
            primary_cluster_size: rs.primary_cluster_size,
            primary_cluster_plv: rs.primary_cluster_plv,
            // Attention spotlight fields (Phase 6C)
            attention_position: as_.primary_position,
            attention_intensity: as_.primary_intensity,
            attention_width: as_.primary_width,
            attention_capacity: as_.attention_capacity,
            secondary_foci_count: as_.secondary_foci_count,
            in_attention_blink: as_.in_blink,
            attention_sampling_effectiveness: as_.sampling_effectiveness,
            in_attention_uptake: as_.in_uptake_phase,
            // Sleep-oscillation fields (Phase 7)
            sleep_stage: sleep.stage,
            time_in_sleep_stage: sleep.time_in_stage_secs,
            sleep_pressure: sleep.sleep_pressure,
            sleep_cycles_completed: sleep.cycles_completed,
            ultradian_cycle_position: sleep.cycle_position,
            consolidation_efficiency: sleep.consolidation_efficiency,
            in_memory_transfer_window: sleep.in_memory_window,
            sleep_consciousness_level: sleep.consciousness_level,
            spindle_count: sleep.spindle_count,
            ripple_count: sleep.ripple_count,
            dream_intensity: sleep.dream_intensity,
            is_sleeping: sleep.is_sleeping,
            delta_power,
            spindle_power,

            // Phase 8: Neuromodulatory dynamics
            dopamine_level: nm_summary.dopamine_level,
            serotonin_level: nm_summary.serotonin_level,
            norepinephrine_level: nm_summary.norepinephrine_level,
            acetylcholine_level: nm_summary.acetylcholine_level,
            dopamine_sensitivity: nm_summary.dopamine_sensitivity,
            serotonin_sensitivity: nm_summary.serotonin_sensitivity,
            mood_valence: nm_summary.mood_valence,
            motivation_drive: nm_summary.motivation_drive,
            learning_rate_mod: nm_summary.learning_rate_mod,
            stress_level: nm_summary.stress_level,

            // Phase 9: Predictive consciousness
            free_energy: self.predictive_consciousness.total_free_energy,
            complexity: self.predictive_consciousness.complexity,
            accuracy: self.predictive_consciousness.accuracy,
            precision_gain: self.predictive_consciousness.precision_gain,
            weighted_prediction_error: self.predictive_consciousness.weighted_prediction_error,
            expected_free_energy: self.predictive_consciousness.expected_free_energy,
            epistemic_value: self.predictive_consciousness.epistemic_value,
            pragmatic_value: self.predictive_consciousness.pragmatic_value,
            exploration_drive: self.predictive_consciousness.exploration_drive,
            surprise: self.predictive_consciousness.surprise,
            sensory_precision: self.predictive_consciousness.sensory.precision,
            perceptual_precision: self.predictive_consciousness.perceptual.precision,
            conceptual_precision: self.predictive_consciousness.conceptual.precision,
            metacognitive_precision: self.predictive_consciousness.metacognitive.precision,
            sensory_prediction: self.predictive_consciousness.sensory.prediction,
            perceptual_prediction: self.predictive_consciousness.perceptual.prediction,
            conceptual_prediction: self.predictive_consciousness.conceptual.prediction,
            metacognitive_prediction: self.predictive_consciousness.metacognitive.prediction,

            // Phase 10: Global workspace
            is_ignited: self.global_workspace.is_ignited,
            broadcasting_processor: self.global_workspace.broadcasting_processor.map(|p| p.name().to_string()),
            broadcast_strength: self.global_workspace.broadcast_strength,
            broadcast_duration: self.global_workspace.broadcast_duration,
            ignition_threshold: self.global_workspace.ignition_threshold,
            time_since_ignition: self.global_workspace.time_since_ignition,
            workspace_integration: self.global_workspace.integration,
            coalition_count: self.global_workspace.active_coalitions.len(),
            perception_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Perception),
            memory_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Memory),
            executive_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Executive),
            emotion_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Emotion),
            motor_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Motor),
            language_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Language),
            metacognition_activation: self.global_workspace.processor_activation(WorkspaceProcessor::Metacognition),
            ignition_count: self.global_workspace.ignition_count,
            total_broadcast_time: self.global_workspace.total_broadcast_time,
        }
    }
}

// Implement the Router trait
impl Router for OscillatoryRouter {
    fn name(&self) -> &'static str {
        "OscillatoryRouter"
    }

    fn current_strategy(&self, phi: f64) -> RoutingStrategy {
        self.predictive_router.current_strategy(phi)
    }

    fn plan(&mut self, state: &LatentConsciousnessState) -> RoutingPlan {
        // Use the phase-locked plan, return the underlying magnitude plan
        let plan = self.plan_phase_locked(state);
        plan.magnitude_plan
    }

    fn execute(&mut self, state: &LatentConsciousnessState, action: ConsciousnessAction) -> RoutingStrategy {
        self.predictive_router.execute_route(state, action)
    }

    fn record_outcome(&mut self, outcome: RoutingOutcome) {
        self.predictive_router.record_outcome(
            outcome.actual_phi,
            outcome.strategy_used,
            outcome.latency_ms,
        );
    }

    fn stats(&self) -> RouterStats {
        let mut base_stats = self.predictive_router.stats();
        base_stats.custom_metrics.insert("phase_lock_accuracy".to_string(), self.stats.phase_lock_accuracy());
        base_stats.custom_metrics.insert("cycles_completed".to_string(), self.stats.cycles_completed as f64);
        base_stats.custom_metrics.insert("avg_coherence".to_string(), self.stats.avg_coherence);
        base_stats
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

/// Summary of oscillatory router state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryRouterSummary {
    // === Core Oscillation State ===
    pub current_phase: OscillatoryPhase,
    pub current_phi: f64,
    pub effective_phi: f64,
    pub coherence: f64,
    pub frequency: f64,
    pub phase_lock_accuracy: f64,
    pub cycles_completed: u64,
    pub pending_operations: usize,
    pub predictive_summary: PredictiveRouterSummary,

    // === Multi-Band Oscillation Fields ===
    pub dominant_band: FrequencyBand,
    pub gamma_power: f64,
    pub theta_power: f64,
    pub alpha_power: f64,
    pub theta_gamma_coupling: f64,
    pub arousal: f64,
    pub attention: f64,
    pub in_memory_window: bool,

    // === Process Resonance Fields (Phase 6C) ===
    /// Overall system resonance (mean PLV across all active process pairs)
    pub system_resonance: f64,
    /// Integration boost factor from resonance (1.0 = no boost, >1 = enhanced)
    pub resonance_integration_boost: f64,
    /// Number of currently resonating process pairs (PLV > threshold)
    pub resonating_pairs: usize,
    /// Number of detected resonant clusters
    pub cluster_count: usize,
    /// Size of the primary (largest) resonant cluster
    pub primary_cluster_size: usize,
    /// Mean PLV of the primary cluster
    pub primary_cluster_plv: f64,

    // === Attention Spotlight Fields (Phase 6C) ===
    /// Position of primary attention focus [0, 1]
    pub attention_position: f64,
    /// Intensity of primary attention focus [0, 1]
    pub attention_intensity: f64,
    /// Width of attention spotlight [0.1, 1.0]
    pub attention_width: f64,
    /// Current attention capacity (decreases with split attention) [0, 1]
    pub attention_capacity: f64,
    /// Number of active secondary attention foci
    pub secondary_foci_count: usize,
    /// Is attention currently in "blink" (refractory) state?
    pub in_attention_blink: bool,
    /// Current attention sampling effectiveness [0.8, 1.0]
    pub attention_sampling_effectiveness: f64,
    /// Is attention in uptake (encoding) phase?
    pub in_attention_uptake: bool,

    // === Sleep-Oscillation Fields (Phase 7) ===
    /// Current sleep stage
    pub sleep_stage: SleepStage,
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
    pub spindle_power: f64,

    // === Neuromodulatory Dynamics Fields (Phase 8) ===
    /// Dopamine level (motivation, reward) [0, 1]
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

    // === Predictive Consciousness Fields (Phase 9) ===
    /// Total free energy (variational free energy)
    pub free_energy: f64,
    /// Complexity term (divergence from prior)
    pub complexity: f64,
    /// Accuracy term (prediction error)
    pub accuracy: f64,
    /// Global precision gain (influenced by arousal)
    pub precision_gain: f64,
    /// Weighted prediction error across levels
    pub weighted_prediction_error: f64,
    /// Expected free energy for current strategy
    pub expected_free_energy: f64,
    /// Epistemic value (information gain potential)
    pub epistemic_value: f64,
    /// Pragmatic value (goal-alignment)
    pub pragmatic_value: f64,
    /// Exploration drive [0=exploit, 1=explore]
    pub exploration_drive: f64,
    /// Surprise (negative log probability)
    pub surprise: f64,
    /// Sensory level precision
    pub sensory_precision: f64,
    /// Perceptual level precision
    pub perceptual_precision: f64,
    /// Conceptual level precision
    pub conceptual_precision: f64,
    /// Metacognitive level precision
    pub metacognitive_precision: f64,
    /// Sensory level prediction
    pub sensory_prediction: f64,
    /// Perceptual level prediction
    pub perceptual_prediction: f64,
    /// Conceptual level prediction
    pub conceptual_prediction: f64,
    /// Metacognitive level prediction
    pub metacognitive_prediction: f64,

    // === Global Workspace Fields (Phase 10) ===
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
    pub ignition_count: u64,
    /// Total broadcast time (seconds)
    pub total_broadcast_time: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillatory_router_creation() {
        let router = OscillatoryRouter::new(OscillatoryRouterConfig::default());
        assert_eq!(router.stats.decisions_made, 0);
        assert_eq!(router.oscillatory_state.frequency, 40.0);
    }

    #[test]
    fn test_oscillatory_router_config_default() {
        let config = OscillatoryRouterConfig::default();
        assert_eq!(config.base_frequency, 40.0);
        assert!(config.phase_locked_scheduling);
    }

    #[test]
    fn test_oscillatory_router_stats_accuracy() {
        let mut stats = OscillatoryRouterStats::default();
        assert_eq!(stats.phase_lock_accuracy(), 0.0);

        stats.decisions_made = 10;
        stats.phase_locked_hits = 8;
        assert!((stats.phase_lock_accuracy() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_phase_window() {
        let window = PhaseWindow {
            target_phase: OscillatoryPhase::Peak,
            required_mode: ProcessingMode::Integration,
            window_fraction: 0.25,
            priority: 0.8,
        };

        let peak_state = OscillatoryState::new(0.0, 40.0, 0.3, 0.7);
        let trough_state = OscillatoryState::new(PI, 40.0, 0.3, 0.7);

        assert!(window.is_active(&peak_state));
        assert!(!window.is_active(&trough_state));
    }
}
