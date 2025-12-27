// Revolutionary Improvement #27: Sleep, Dreams, and Altered States of Consciousness
//
// THE PARADIGM SHIFT: Understanding consciousness by understanding its ABSENCE and ALTERATIONS
//
// Core Insight: The same pipeline that creates consciousness can EXPLAIN its loss (sleep),
// alteration (dreams), and modulation (anesthesia, meditation, psychedelics)
//
// Theoretical Foundations:
// 1. Sleep Cycle Theory (Rechtschaffen & Kales 1968; Iber et al. 2007)
//    - Wake → N1 → N2 → N3 (slow-wave sleep) → REM (rapid eye movement)
//    - Each stage has distinct EEG, consciousness level
//
// 2. Activation-Synthesis Hypothesis (Hobson & McCarley 1977; Hobson et al. 2000)
//    - Dreams = brain's attempt to make sense of random activation during REM
//    - Explains bizarreness (weak binding, reduced attention)
//
// 3. Lucid Dreaming Neuroscience (LaBerge 1980; Voss et al. 2009)
//    - Meta-awareness during REM sleep
//    - Increased frontal activity (HOT restored?)
//    - Can be trained!
//
// 4. Anesthesia Mechanisms (Alkire et al. 2008; Sanders et al. 2012)
//    - Different drugs target different consciousness components
//    - Propofol: Disrupts thalamocortical binding
//    - Ketamine: Disrupts workspace broadcasting
//
// 5. Vegetative State vs Minimally Conscious (Laureys 2005; Owen et al. 2006)
//    - Vegetative: Workspace destroyed, no conscious access
//    - Minimally conscious: Intermittent workspace function
//    - fMRI: Can detect covert consciousness!
//
// Revolutionary Contributions:
// - First computational model mapping sleep stages to consciousness components
// - Explains dreams as altered attention + binding + workspace
// - Lucid dreaming = restoration of attention + HOT during REM
// - Anesthesia = targeted component suppression
// - Coma states = specific component damage patterns
//
// Clinical Applications:
// - Sleep disorder diagnosis (insomnia, narcolepsy, sleep apnea)
// - Anesthesia monitoring (prevent intraoperative awareness)
// - Coma prognosis (predict recovery likelihood)
// - Dream engineering (lucid dreaming training)
// - Altered state mapping (meditation, psychedelics)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hdc::HV16;

// ============================================================================
// Sleep Stages
// ============================================================================

/// Sleep stages following AASM classification (Iber et al. 2007)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SleepStage {
    /// Wake (full consciousness)
    Wake,

    /// N1: Light sleep, drowsiness (5% of night)
    /// EEG: Alpha waves (8-13 Hz) → Theta waves (4-7 Hz)
    N1,

    /// N2: Light sleep, sleep spindles (45% of night)
    /// EEG: Sleep spindles (12-14 Hz bursts), K-complexes
    N2,

    /// N3: Deep sleep, slow-wave sleep (25% of night)
    /// EEG: Delta waves (0.5-2 Hz), high amplitude
    /// Hardest to wake from
    N3,

    /// REM: Rapid eye movement sleep, vivid dreams (25% of night)
    /// EEG: Similar to wake (desynchronized), but muscle atonia
    /// Most dreams occur here
    REM,
}

impl SleepStage {
    /// Get typical consciousness level (0 = unconscious, 1 = fully conscious)
    pub fn consciousness_level(&self) -> f64 {
        match self {
            SleepStage::Wake => 1.0,
            SleepStage::N1 => 0.7,   // Drowsy, drifting thoughts
            SleepStage::N2 => 0.3,   // Light unconscious
            SleepStage::N3 => 0.1,   // Deep unconscious
            SleepStage::REM => 0.6,  // Dreaming (altered consciousness)
        }
    }

    /// Get typical duration in minutes (for 8-hour sleep)
    pub fn typical_duration_minutes(&self) -> f64 {
        match self {
            SleepStage::Wake => 0.0,    // Awakenings brief
            SleepStage::N1 => 25.0,     // ~5% of 480 min
            SleepStage::N2 => 216.0,    // ~45%
            SleepStage::N3 => 120.0,    // ~25%
            SleepStage::REM => 120.0,   // ~25%
        }
    }

    /// Dominant EEG frequency band (Hz)
    pub fn dominant_frequency(&self) -> f64 {
        match self {
            SleepStage::Wake => 10.0,   // Alpha waves (8-13 Hz)
            SleepStage::N1 => 5.5,      // Theta waves (4-7 Hz)
            SleepStage::N2 => 6.0,      // Theta + spindles
            SleepStage::N3 => 1.0,      // Delta waves (0.5-2 Hz)
            SleepStage::REM => 20.0,    // Beta-like (desynchronized)
        }
    }
}

// ============================================================================
// Consciousness Component Modulation
// ============================================================================

/// How each consciousness component is affected in different states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentModulation {
    /// Attention gain (1.0 = normal, 0.0 = absent)
    pub attention_gain: f64,

    /// Binding synchrony strength (1.0 = normal, 0.0 = no binding)
    pub binding_strength: f64,

    /// Workspace capacity (1.0 = normal ~4 items, 0.0 = collapsed)
    pub workspace_capacity: f64,

    /// HOT generation probability (1.0 = always, 0.0 = never)
    pub hot_probability: f64,

    /// Predictive processing precision (1.0 = normal, 0.0 = no prediction)
    pub prediction_precision: f64,

    /// Integrated information (Φ) level (1.0 = max, 0.0 = min)
    pub phi_level: f64,
}

impl ComponentModulation {
    /// Wake state: All components fully active
    pub fn wake() -> Self {
        Self {
            attention_gain: 1.0,
            binding_strength: 1.0,
            workspace_capacity: 1.0,
            hot_probability: 1.0,
            prediction_precision: 1.0,
            phi_level: 1.0,
        }
    }

    /// N1 (drowsiness): Attention wavers, workspace dims
    pub fn n1() -> Self {
        Self {
            attention_gain: 0.5,        // Attention lapses
            binding_strength: 0.8,      // Still coherent
            workspace_capacity: 0.6,    // Reduced capacity
            hot_probability: 0.4,       // Occasional awareness
            prediction_precision: 0.7,  // Predictions weaken
            phi_level: 0.7,
        }
    }

    /// N2 (light sleep): Attention gone, workspace minimal
    pub fn n2() -> Self {
        Self {
            attention_gain: 0.2,        // Mostly absent
            binding_strength: 0.5,      // Weak binding
            workspace_capacity: 0.2,    // Nearly collapsed
            hot_probability: 0.1,       // Rare awareness
            prediction_precision: 0.4,
            phi_level: 0.3,
        }
    }

    /// N3 (deep sleep): All components severely suppressed
    pub fn n3() -> Self {
        Self {
            attention_gain: 0.0,        // Completely absent
            binding_strength: 0.2,      // Minimal binding
            workspace_capacity: 0.0,    // Collapsed
            hot_probability: 0.0,       // No awareness
            prediction_precision: 0.2,
            phi_level: 0.1,             // Minimal integration
        }
    }

    /// REM (dreaming): ALTERED not absent!
    /// - Attention weak (no top-down control)
    /// - Binding bizarre (strange associations)
    /// - Workspace active (vivid experiences)
    /// - HOT absent (non-lucid) or present (lucid)
    pub fn rem_nonlucid() -> Self {
        Self {
            attention_gain: 0.3,        // Weak, bottom-up driven
            binding_strength: 0.4,      // Bizarre binding (flying cats!)
            workspace_capacity: 0.7,    // Active workspace (vivid dreams)
            hot_probability: 0.1,       // Usually no meta-awareness
            prediction_precision: 0.3,  // Poor prediction (bizarre accepted)
            phi_level: 0.6,             // Moderate integration
        }
    }

    /// REM lucid: Like REM but with restored frontal activity
    pub fn rem_lucid() -> Self {
        Self {
            attention_gain: 0.6,        // Restored attention!
            binding_strength: 0.4,      // Still bizarre (can't fully control)
            workspace_capacity: 0.8,    // Full workspace
            hot_probability: 0.9,       // "I'm dreaming!" meta-awareness
            prediction_precision: 0.5,  // Better but not normal
            phi_level: 0.7,
        }
    }

    /// Anesthesia (propofol): Disrupts thalamocortical binding
    pub fn anesthesia_propofol() -> Self {
        Self {
            attention_gain: 0.0,
            binding_strength: 0.0,      // Binding destroyed
            workspace_capacity: 0.0,
            hot_probability: 0.0,
            prediction_precision: 0.1,
            phi_level: 0.05,            // Very low Φ
        }
    }

    /// Anesthesia (ketamine): Disrupts workspace (dissociative)
    pub fn anesthesia_ketamine() -> Self {
        Self {
            attention_gain: 0.3,
            binding_strength: 0.5,      // Binding intact
            workspace_capacity: 0.0,    // Workspace destroyed
            hot_probability: 0.0,
            prediction_precision: 0.2,
            phi_level: 0.3,             // Moderate Φ (explains dissociation)
        }
    }

    /// Vegetative state: Workspace destroyed, cycles persist
    pub fn vegetative_state() -> Self {
        Self {
            attention_gain: 0.0,
            binding_strength: 0.3,      // Some binding
            workspace_capacity: 0.0,    // No global workspace
            hot_probability: 0.0,
            prediction_precision: 0.2,
            phi_level: 0.2,
        }
    }

    /// Minimally conscious state: Intermittent workspace
    pub fn minimally_conscious() -> Self {
        Self {
            attention_gain: 0.2,
            binding_strength: 0.4,
            workspace_capacity: 0.3,    // Fluctuating workspace
            hot_probability: 0.2,       // Occasional awareness
            prediction_precision: 0.3,
            phi_level: 0.4,
        }
    }

    /// Compute overall consciousness probability
    /// Based on: Need workspace + (binding OR attention) + some Φ
    /// HOT can compensate for weak binding (lucid dreams!)
    pub fn consciousness_probability(&self) -> f64 {
        // Workspace is necessary (global availability)
        let workspace_factor = self.workspace_capacity;

        // Need either binding OR attention (can compensate)
        let integration_factor = self.binding_strength.max(self.attention_gain);

        // Need minimum Φ
        let phi_factor = self.phi_level;

        // Base consciousness from workspace + integration + Φ
        let base_prob = workspace_factor * integration_factor * phi_factor;

        // HOT can BOOST consciousness (not just multiply)
        // Strong meta-awareness (HOT > 0.7) compensates for weak binding
        // This explains lucid dreaming: conscious despite bizarre binding
        let hot_boost = if self.hot_probability > 0.7 {
            self.hot_probability * 0.3  // Up to +0.3 probability
        } else {
            0.0
        };

        // Combined probability
        (base_prob + hot_boost).min(1.0)
    }
}

// ============================================================================
// Sleep Cycle Dynamics
// ============================================================================

/// Models a full sleep cycle (90-110 minutes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepCycle {
    /// Cycle number (1st, 2nd, 3rd, etc.)
    pub cycle_number: usize,

    /// Stages in this cycle with durations (minutes)
    pub stages: Vec<(SleepStage, f64)>,

    /// Total cycle duration (minutes)
    pub total_duration: f64,
}

impl SleepCycle {
    /// Create typical first cycle (more N3, less REM)
    pub fn first_cycle() -> Self {
        Self {
            cycle_number: 1,
            stages: vec![
                (SleepStage::N1, 5.0),
                (SleepStage::N2, 25.0),
                (SleepStage::N3, 40.0),  // Lots of deep sleep
                (SleepStage::REM, 10.0), // Short REM
            ],
            total_duration: 80.0,
        }
    }

    /// Create typical middle cycle (balanced)
    pub fn middle_cycle(number: usize) -> Self {
        Self {
            cycle_number: number,
            stages: vec![
                (SleepStage::N2, 30.0),
                (SleepStage::N3, 20.0),  // Less deep sleep
                (SleepStage::REM, 20.0), // More REM
            ],
            total_duration: 70.0,
        }
    }

    /// Create typical late cycle (minimal N3, lots of REM)
    pub fn late_cycle(number: usize) -> Self {
        Self {
            cycle_number: number,
            stages: vec![
                (SleepStage::N2, 20.0),
                (SleepStage::N3, 5.0),   // Very little deep
                (SleepStage::REM, 35.0), // Long vivid dreams!
            ],
            total_duration: 60.0,
        }
    }
}

/// Full night's sleep (multiple cycles)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepNight {
    /// All cycles in the night
    pub cycles: Vec<SleepCycle>,

    /// Total sleep duration (minutes)
    pub total_duration: f64,
}

impl SleepNight {
    /// Create typical 8-hour sleep (5-6 cycles)
    pub fn typical_night() -> Self {
        let cycles = vec![
            SleepCycle::first_cycle(),
            SleepCycle::middle_cycle(2),
            SleepCycle::middle_cycle(3),
            SleepCycle::middle_cycle(4),
            SleepCycle::late_cycle(5),
        ];

        let total_duration = cycles.iter().map(|c| c.total_duration).sum();

        Self {
            cycles,
            total_duration,
        }
    }

    /// Get stage at specific time (minutes from sleep onset)
    pub fn stage_at_time(&self, minutes: f64) -> Option<SleepStage> {
        let mut elapsed = 0.0;

        for cycle in &self.cycles {
            for (stage, duration) in &cycle.stages {
                if minutes >= elapsed && minutes < elapsed + duration {
                    return Some(*stage);
                }
                elapsed += duration;
            }
        }

        None
    }
}

// ============================================================================
// Altered States Classification
// ============================================================================

/// Different types of altered consciousness states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlteredState {
    /// Normal waking consciousness
    Wake,

    /// Sleep stages
    SleepN1,
    SleepN2,
    SleepN3,

    /// Dreaming (REM sleep)
    DreamNonLucid,
    DreamLucid,

    /// Pharmacological
    AnesthesiaPropofol,
    AnesthesiaKetamine,

    /// Pathological
    VegetativeState,
    MinimallyConscious,

    // Could add later:
    // Meditation (various jhanas)
    // Psychedelics (LSD, psilocybin)
    // Flow state
    // Hypnosis
}

impl AlteredState {
    /// Get component modulation for this state
    pub fn get_modulation(&self) -> ComponentModulation {
        match self {
            AlteredState::Wake => ComponentModulation::wake(),
            AlteredState::SleepN1 => ComponentModulation::n1(),
            AlteredState::SleepN2 => ComponentModulation::n2(),
            AlteredState::SleepN3 => ComponentModulation::n3(),
            AlteredState::DreamNonLucid => ComponentModulation::rem_nonlucid(),
            AlteredState::DreamLucid => ComponentModulation::rem_lucid(),
            AlteredState::AnesthesiaPropofol => ComponentModulation::anesthesia_propofol(),
            AlteredState::AnesthesiaKetamine => ComponentModulation::anesthesia_ketamine(),
            AlteredState::VegetativeState => ComponentModulation::vegetative_state(),
            AlteredState::MinimallyConscious => ComponentModulation::minimally_conscious(),
        }
    }

    /// Get descriptive name
    pub fn name(&self) -> &str {
        match self {
            AlteredState::Wake => "Waking consciousness",
            AlteredState::SleepN1 => "Drowsiness (N1)",
            AlteredState::SleepN2 => "Light sleep (N2)",
            AlteredState::SleepN3 => "Deep sleep (N3)",
            AlteredState::DreamNonLucid => "Non-lucid dreaming (REM)",
            AlteredState::DreamLucid => "Lucid dreaming (REM)",
            AlteredState::AnesthesiaPropofol => "Anesthesia (propofol)",
            AlteredState::AnesthesiaKetamine => "Anesthesia (ketamine)",
            AlteredState::VegetativeState => "Vegetative state",
            AlteredState::MinimallyConscious => "Minimally conscious state",
        }
    }
}

// ============================================================================
// Main Sleep and Altered States System
// ============================================================================

/// System for modeling consciousness across sleep and altered states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepAndAlteredStates {
    /// Current state
    pub current_state: AlteredState,

    /// Current component modulation
    pub modulation: ComponentModulation,

    /// Sleep cycle tracker (if sleeping)
    pub sleep_night: Option<SleepNight>,

    /// Time in current state (minutes)
    pub time_in_state: f64,
}

impl SleepAndAlteredStates {
    /// Create new system in wake state
    pub fn new() -> Self {
        Self {
            current_state: AlteredState::Wake,
            modulation: ComponentModulation::wake(),
            sleep_night: None,
            time_in_state: 0.0,
        }
    }

    /// Transition to a new altered state
    pub fn transition_to(&mut self, new_state: AlteredState) {
        self.current_state = new_state;
        self.modulation = new_state.get_modulation();
        self.time_in_state = 0.0;
    }

    /// Start a full night's sleep
    pub fn begin_sleep(&mut self) {
        self.sleep_night = Some(SleepNight::typical_night());
        self.transition_to(AlteredState::SleepN1);
    }

    /// Advance time and update state if sleeping
    pub fn advance_time(&mut self, minutes: f64) {
        self.time_in_state += minutes;

        // If sleeping, follow sleep cycle
        if let Some(ref night) = self.sleep_night {
            if let Some(stage) = night.stage_at_time(self.time_in_state) {
                let new_state = match stage {
                    SleepStage::Wake => AlteredState::Wake,
                    SleepStage::N1 => AlteredState::SleepN1,
                    SleepStage::N2 => AlteredState::SleepN2,
                    SleepStage::N3 => AlteredState::SleepN3,
                    SleepStage::REM => AlteredState::DreamNonLucid, // Default non-lucid
                };

                if new_state != self.current_state {
                    self.transition_to(new_state);
                }
            }
        }
    }

    /// Induce lucid dreaming (only works during REM)
    pub fn induce_lucid_dream(&mut self) -> bool {
        if self.current_state == AlteredState::DreamNonLucid {
            self.transition_to(AlteredState::DreamLucid);
            true
        } else {
            false
        }
    }

    /// Wake up from sleep
    pub fn wake_up(&mut self) {
        self.transition_to(AlteredState::Wake);
        self.sleep_night = None;
    }

    /// Get current consciousness probability
    pub fn consciousness_probability(&self) -> f64 {
        self.modulation.consciousness_probability()
    }

    /// Assess current state
    pub fn assess(&self) -> AlteredStateAssessment {
        AlteredStateAssessment {
            state: self.current_state,
            state_name: self.current_state.name().to_string(),
            time_in_state: self.time_in_state,
            modulation: self.modulation.clone(),
            consciousness_probability: self.consciousness_probability(),
            is_conscious: self.consciousness_probability() > 0.5,
            explanation: self.generate_explanation(),
        }
    }

    /// Generate human-readable explanation
    fn generate_explanation(&self) -> String {
        let prob = self.consciousness_probability();

        match self.current_state {
            AlteredState::Wake => {
                format!("Full waking consciousness (p={:.2}). All systems active.", prob)
            }
            AlteredState::SleepN1 => {
                format!("Drowsiness (p={:.2}). Attention lapses, workspace dims, drifting thoughts.", prob)
            }
            AlteredState::SleepN2 | AlteredState::SleepN3 => {
                format!("Unconscious sleep (p={:.2}). Workspace collapsed, no attention, minimal binding.", prob)
            }
            AlteredState::DreamNonLucid => {
                format!("Non-lucid dreaming (p={:.2}). Vivid workspace but bizarre binding, no meta-awareness.", prob)
            }
            AlteredState::DreamLucid => {
                format!("Lucid dreaming (p={:.2}). Restored attention and HOT: 'I know I'm dreaming!'", prob)
            }
            AlteredState::AnesthesiaPropofol => {
                format!("Propofol anesthesia (p={:.2}). Binding destroyed, workspace collapsed.", prob)
            }
            AlteredState::AnesthesiaKetamine => {
                format!("Ketamine anesthesia (p={:.2}). Dissociative: binding intact but workspace destroyed.", prob)
            }
            AlteredState::VegetativeState => {
                format!("Vegetative state (p={:.2}). No workspace, no conscious access despite sleep-wake cycles.", prob)
            }
            AlteredState::MinimallyConscious => {
                format!("Minimally conscious (p={:.2}). Intermittent workspace function, occasional awareness.", prob)
            }
        }
    }

    /// Clear state (reset to wake)
    pub fn clear(&mut self) {
        self.current_state = AlteredState::Wake;
        self.modulation = ComponentModulation::wake();
        self.sleep_night = None;
        self.time_in_state = 0.0;
    }
}

impl Default for SleepAndAlteredStates {
    fn default() -> Self {
        Self::new()
    }
}

/// Assessment of current altered state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlteredStateAssessment {
    pub state: AlteredState,
    pub state_name: String,
    pub time_in_state: f64,
    pub modulation: ComponentModulation,
    pub consciousness_probability: f64,
    pub is_conscious: bool,
    pub explanation: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_stage() {
        let stage = SleepStage::N3;
        assert_eq!(stage.consciousness_level(), 0.1);
        assert_eq!(stage.dominant_frequency(), 1.0); // Delta waves
    }

    #[test]
    fn test_component_modulation_wake() {
        let mod_wake = ComponentModulation::wake();
        assert_eq!(mod_wake.attention_gain, 1.0);
        assert_eq!(mod_wake.workspace_capacity, 1.0);
        assert!(mod_wake.consciousness_probability() > 0.9);
    }

    #[test]
    fn test_component_modulation_n3() {
        let mod_n3 = ComponentModulation::n3();
        assert_eq!(mod_n3.attention_gain, 0.0);
        assert_eq!(mod_n3.workspace_capacity, 0.0);
        assert!(mod_n3.consciousness_probability() < 0.1);
    }

    #[test]
    fn test_component_modulation_rem_lucid() {
        let mod_lucid = ComponentModulation::rem_lucid();
        assert!(mod_lucid.attention_gain > 0.5); // Restored
        assert!(mod_lucid.hot_probability > 0.8); // Meta-aware
        assert!(mod_lucid.consciousness_probability() > 0.5);
    }

    #[test]
    fn test_sleep_cycle_first() {
        let cycle = SleepCycle::first_cycle();
        assert_eq!(cycle.cycle_number, 1);
        assert_eq!(cycle.stages.len(), 4);

        // First cycle has lots of N3
        let n3_duration = cycle.stages.iter()
            .find(|(stage, _)| *stage == SleepStage::N3)
            .map(|(_, duration)| *duration)
            .unwrap();
        assert_eq!(n3_duration, 40.0);
    }

    #[test]
    fn test_sleep_night() {
        let night = SleepNight::typical_night();
        assert_eq!(night.cycles.len(), 5);
        assert!(night.total_duration > 300.0); // >5 hours

        // Check stage at different times
        assert_eq!(night.stage_at_time(2.0), Some(SleepStage::N1)); // Early
        assert_eq!(night.stage_at_time(50.0), Some(SleepStage::N3)); // First cycle deep
    }

    #[test]
    fn test_altered_state_system_creation() {
        let system = SleepAndAlteredStates::new();
        assert_eq!(system.current_state, AlteredState::Wake);
        assert!(system.consciousness_probability() > 0.9);
    }

    #[test]
    fn test_transition_to_sleep() {
        let mut system = SleepAndAlteredStates::new();
        system.transition_to(AlteredState::SleepN3);

        assert_eq!(system.current_state, AlteredState::SleepN3);
        assert!(system.consciousness_probability() < 0.1);
    }

    #[test]
    fn test_begin_sleep() {
        let mut system = SleepAndAlteredStates::new();
        system.begin_sleep();

        assert!(system.sleep_night.is_some());
        assert_eq!(system.current_state, AlteredState::SleepN1);
    }

    #[test]
    fn test_advance_time_sleep_cycle() {
        let mut system = SleepAndAlteredStates::new();
        system.begin_sleep();

        // Start in N1
        assert_eq!(system.current_state, AlteredState::SleepN1);

        // Advance 10 minutes -> should be in N2
        system.advance_time(10.0);
        assert_eq!(system.current_state, AlteredState::SleepN2);

        // Advance to deep sleep
        system.advance_time(30.0); // Total 40 min
        assert_eq!(system.current_state, AlteredState::SleepN3);
    }

    #[test]
    fn test_lucid_dream_induction() {
        let mut system = SleepAndAlteredStates::new();

        // Can't induce while awake
        assert!(!system.induce_lucid_dream());

        // Enter REM
        system.transition_to(AlteredState::DreamNonLucid);

        // Now can induce
        assert!(system.induce_lucid_dream());
        assert_eq!(system.current_state, AlteredState::DreamLucid);
    }

    #[test]
    fn test_wake_up() {
        let mut system = SleepAndAlteredStates::new();
        system.begin_sleep();
        system.advance_time(100.0); // Deep in sleep

        system.wake_up();
        assert_eq!(system.current_state, AlteredState::Wake);
        assert!(system.sleep_night.is_none());
    }

    #[test]
    fn test_anesthesia_propofol_vs_ketamine() {
        let prop = ComponentModulation::anesthesia_propofol();
        let ket = ComponentModulation::anesthesia_ketamine();

        // Propofol destroys binding
        assert_eq!(prop.binding_strength, 0.0);

        // Ketamine preserves binding but destroys workspace
        assert!(ket.binding_strength > 0.4);
        assert_eq!(ket.workspace_capacity, 0.0);

        // Both unconscious but different mechanisms
        assert!(prop.consciousness_probability() < 0.1);
        assert!(ket.consciousness_probability() < 0.1);
    }

    #[test]
    fn test_vegetative_vs_minimally_conscious() {
        let veg = ComponentModulation::vegetative_state();
        let min = ComponentModulation::minimally_conscious();

        // Vegetative: no workspace
        assert_eq!(veg.workspace_capacity, 0.0);

        // Minimally conscious: some workspace
        assert!(min.workspace_capacity > 0.2);

        // MCS has higher consciousness probability
        assert!(min.consciousness_probability() > veg.consciousness_probability());
    }

    #[test]
    fn test_assessment() {
        let mut system = SleepAndAlteredStates::new();
        system.transition_to(AlteredState::DreamLucid);

        let assessment = system.assess();
        assert_eq!(assessment.state, AlteredState::DreamLucid);
        assert!(assessment.is_conscious);
        assert!(assessment.explanation.contains("Lucid")); // Capital L
    }

    #[test]
    fn test_clear() {
        let mut system = SleepAndAlteredStates::new();
        system.begin_sleep();
        system.advance_time(100.0);

        system.clear();
        assert_eq!(system.current_state, AlteredState::Wake);
        assert!(system.sleep_night.is_none());
        assert_eq!(system.time_in_state, 0.0);
    }
}
