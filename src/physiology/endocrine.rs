//! Endocrine System - The Chemical Layer
//!
//! Week 4 Days 1-3: The Endocrine Core (Moods)
//!
//! The endocrine system is the "chemical layer" underneath the "neural layer".
//! It implements slow-moving hormones that create persistent moods and regulate
//! the entire cognitive system.
//!
//! # Paradigm Shift: Intelligence is Regulation
//!
//! Traditional AI: Intelligence = Computation
//! Sophia: Intelligence = Regulation of Chemical State
//!
//! # The Three Core Hormones
//!
//! 1. **Cortisol** (Stress Hormone)
//!    - High = Paranoid, risk-averse, defensive
//!    - Low = Calm, exploratory, open
//!    - Triggered by: Errors, threats, safety violations
//!
//! 2. **Dopamine** (Reward Hormone)
//!    - High = Reward-seeking, exploratory, optimistic
//!    - Low = Conservative, pessimistic, cautious
//!    - Triggered by: Success, positive feedback, learning
//!
//! 3. **Acetylcholine** (Focus Hormone)
//!    - High = Narrow attention, deep focus, tunnel vision
//!    - Low = Broad attention, scanning, exploration
//!    - Triggered by: Sustained focus, deep work, flow state
//!
//! # Design Principles
//!
//! - **Slow Decay**: Hormones decay over minutes, not milliseconds
//! - **Global Modulation**: Hormones affect all brain processes
//! - **Event-Driven**: Specific events trigger hormone release
//! - **Homeostasis**: System seeks balance, not extremes

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Hormone state - the "chemical milieu" of the brain
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HormoneState {
    /// Cortisol (0.0-1.0): Stress hormone
    /// High = paranoid, risk-averse, defensive
    pub cortisol: f32,

    /// Dopamine (0.0-1.0): Reward hormone
    /// High = reward-seeking, exploratory, optimistic
    pub dopamine: f32,

    /// Acetylcholine (0.0-1.0): Focus hormone
    /// High = narrow attention, deep focus
    pub acetylcholine: f32,
}

impl HormoneState {
    /// Create neutral hormone state (baseline)
    pub fn neutral() -> Self {
        Self {
            cortisol: 0.3,      // Slight baseline vigilance
            dopamine: 0.5,      // Balanced motivation
            acetylcholine: 0.4, // Moderate focus
        }
    }

    /// Create stressed hormone state (high cortisol)
    /// Arousal > 0.6, Valence < 0 = "Anxious"
    pub fn stressed() -> Self {
        Self {
            cortisol: 0.8,      // High stress
            dopamine: 0.3,      // Low reward
            acetylcholine: 0.8, // High focus (stressed attention)
        }
    }

    /// Create excited hormone state (high dopamine)
    /// Arousal > 0.6, Valence > 0 = "Excited"
    pub fn excited() -> Self {
        Self {
            cortisol: 0.2,      // Low stress
            dopamine: 0.9,      // High reward
            acetylcholine: 0.8, // Energized focus
        }
    }

    /// Create focused hormone state (high acetylcholine)
    pub fn focused() -> Self {
        Self {
            cortisol: 0.3,
            dopamine: 0.6,
            acetylcholine: 0.9,
        }
    }

    /// Overall arousal level (0.0-1.0)
    /// High arousal = alert, activated
    /// Low arousal = calm, relaxed
    pub fn arousal(&self) -> f32 {
        (self.cortisol + self.dopamine + self.acetylcholine) / 3.0
    }

    /// Overall valence (pleasure vs displeasure)
    /// Positive = happy, pleasant
    /// Negative = unhappy, unpleasant
    pub fn valence(&self) -> f32 {
        // High dopamine, low cortisol = positive
        self.dopamine - self.cortisol
    }

    /// Mood descriptor based on hormone state
    pub fn mood(&self) -> &'static str {
        let arousal = self.arousal();
        let valence = self.valence();

        match (arousal > 0.6, valence > 0.1) {
            (true, true) => "Excited",     // High arousal, positive valence
            (true, false) => "Anxious",    // High arousal, negative valence
            (false, true) => "Calm",       // Low arousal, positive valence
            (false, false) => "Depressed", // Low arousal, negative valence
        }
    }
}

/// Events that trigger hormone changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HormoneEvent {
    /// Error or failure occurred
    Error { severity: f32 },

    /// Success or positive outcome
    Success { magnitude: f32 },

    /// Threat detected (e.g., safety violation)
    Threat { intensity: f32 },

    /// Sustained focus period
    DeepFocus { duration_cycles: u32 },

    /// Reward received
    Reward { value: f32 },

    /// Context switch (disrupts focus)
    ContextSwitch,

    /// Recovery period (rest)
    Recovery,
}

/// Configuration for endocrine system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndocrineConfig {
    /// Decay rate for cortisol (per cycle)
    pub cortisol_decay_rate: f32,

    /// Decay rate for dopamine (per cycle)
    pub dopamine_decay_rate: f32,

    /// Decay rate for acetylcholine (per cycle)
    pub acetylcholine_decay_rate: f32,

    /// Cortisol response to errors (0.0-1.0)
    pub cortisol_sensitivity: f32,

    /// Dopamine response to success (0.0-1.0)
    pub dopamine_sensitivity: f32,

    /// Acetylcholine response to focus (0.0-1.0)
    pub acetylcholine_sensitivity: f32,

    /// Maximum hormone level (clamp)
    pub max_hormone_level: f32,

    /// Minimum hormone level (floor)
    pub min_hormone_level: f32,

    /// History window size (for trend analysis)
    pub history_window: usize,
}

impl Default for EndocrineConfig {
    fn default() -> Self {
        Self {
            cortisol_decay_rate: 0.02,       // Slow decay (~50 cycles to baseline)
            dopamine_decay_rate: 0.05,       // Faster decay (~20 cycles to baseline)
            acetylcholine_decay_rate: 0.03,  // Medium decay (~33 cycles to baseline)
            cortisol_sensitivity: 0.3,
            dopamine_sensitivity: 0.2,
            acetylcholine_sensitivity: 0.15,
            max_hormone_level: 1.0,
            min_hormone_level: 0.0,
            history_window: 100,
        }
    }
}

/// The endocrine system - regulates mood and arousal
#[derive(Debug)]
pub struct EndocrineSystem {
    /// Current hormone state
    state: HormoneState,

    /// Configuration
    config: EndocrineConfig,

    /// History of hormone states (for trend analysis)
    history: VecDeque<HormoneState>,

    /// Cycle counter
    cycle_count: u64,

    /// Focus streak counter (consecutive focus cycles)
    focus_streak: u32,
}

impl EndocrineSystem {
    /// Create new endocrine system with neutral state
    pub fn new(config: EndocrineConfig) -> Self {
        Self {
            state: HormoneState::neutral(),
            config,
            history: VecDeque::with_capacity(100),
            cycle_count: 0,
            focus_streak: 0,
        }
    }

    /// Get current hormone state (read-only)
    pub fn state(&self) -> &HormoneState {
        &self.state
    }

    /// Get configuration
    pub fn config(&self) -> &EndocrineConfig {
        &self.config
    }

    /// Process hormone event and update state
    pub fn process_event(&mut self, event: HormoneEvent) {
        match event {
            HormoneEvent::Error { severity } => {
                // Increase cortisol (stress response)
                let delta = severity * self.config.cortisol_sensitivity;
                self.state.cortisol = (self.state.cortisol + delta)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);

                // Decrease dopamine (negative feedback)
                self.state.dopamine = (self.state.dopamine - delta * 0.5)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);

                // Reset focus streak
                self.focus_streak = 0;
            }

            HormoneEvent::Success { magnitude } => {
                // Increase dopamine (reward)
                let delta = magnitude * self.config.dopamine_sensitivity;
                self.state.dopamine = (self.state.dopamine + delta)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);

                // Decrease cortisol (stress relief)
                self.state.cortisol = (self.state.cortisol - delta * 0.3)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);
            }

            HormoneEvent::Threat { intensity } => {
                // Spike cortisol (immediate stress response)
                let delta = intensity * self.config.cortisol_sensitivity * 1.5;
                self.state.cortisol = (self.state.cortisol + delta)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);

                // Reset focus
                self.focus_streak = 0;
                self.state.acetylcholine = (self.state.acetylcholine - 0.2)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);
            }

            HormoneEvent::DeepFocus { duration_cycles } => {
                // Increase acetylcholine (focus building)
                self.focus_streak += duration_cycles;
                let delta = (duration_cycles as f32 * 0.1) * self.config.acetylcholine_sensitivity;
                self.state.acetylcholine = (self.state.acetylcholine + delta)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);

                // Slight dopamine increase (flow state reward)
                if self.focus_streak > 5 {
                    self.state.dopamine = (self.state.dopamine + 0.05)
                        .clamp(self.config.min_hormone_level, self.config.max_hormone_level);
                }
            }

            HormoneEvent::Reward { value } => {
                // Dopamine spike (learning signal)
                let delta = value * self.config.dopamine_sensitivity * 1.2;
                self.state.dopamine = (self.state.dopamine + delta)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);
            }

            HormoneEvent::ContextSwitch => {
                // Disrupts focus
                self.focus_streak = 0;
                self.state.acetylcholine = (self.state.acetylcholine - 0.15)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);

                // Slight stress from disruption
                self.state.cortisol = (self.state.cortisol + 0.1)
                    .clamp(self.config.min_hormone_level, self.config.max_hormone_level);
            }

            HormoneEvent::Recovery => {
                // Accelerate return to baseline
                self.decay_hormones_fast();
            }
        }
    }

    /// Natural decay of hormones (call every cycle)
    pub fn decay_cycle(&mut self) {
        self.cycle_count += 1;

        // Decay towards baseline
        let cortisol_baseline = 0.3;
        let dopamine_baseline = 0.5;
        let acetylcholine_baseline = 0.4;

        // Move towards baseline
        self.state.cortisol = Self::decay_towards(
            self.state.cortisol,
            cortisol_baseline,
            self.config.cortisol_decay_rate,
        );

        self.state.dopamine = Self::decay_towards(
            self.state.dopamine,
            dopamine_baseline,
            self.config.dopamine_decay_rate,
        );

        self.state.acetylcholine = Self::decay_towards(
            self.state.acetylcholine,
            acetylcholine_baseline,
            self.config.acetylcholine_decay_rate,
        );

        // Record history
        if self.history.len() >= self.config.history_window {
            self.history.pop_front();
        }
        self.history.push_back(self.state.clone());
    }

    /// Decay towards baseline
    fn decay_towards(current: f32, baseline: f32, rate: f32) -> f32 {
        if current > baseline {
            (current - rate).max(baseline)
        } else {
            (current + rate).min(baseline)
        }
    }

    /// Fast decay (for recovery periods)
    fn decay_hormones_fast(&mut self) {
        let fast_rate = 0.1; // 10x normal decay
        self.state.cortisol = Self::decay_towards(self.state.cortisol, 0.3, fast_rate);
        self.state.dopamine = Self::decay_towards(self.state.dopamine, 0.5, fast_rate);
        self.state.acetylcholine = Self::decay_towards(self.state.acetylcholine, 0.4, fast_rate);
    }

    /// Get hormone trend (increasing/decreasing/stable)
    pub fn get_trend(&self, window: usize) -> HormoneTrend {
        if self.history.len() < window {
            return HormoneTrend {
                cortisol: Trend::Stable,
                dopamine: Trend::Stable,
                acetylcholine: Trend::Stable,
            };
        }

        let recent: Vec<_> = self.history.iter().rev().take(window).collect();

        let cortisol_trend = Self::calculate_trend(recent.iter().map(|s| s.cortisol));
        let dopamine_trend = Self::calculate_trend(recent.iter().map(|s| s.dopamine));
        let acetylcholine_trend = Self::calculate_trend(recent.iter().map(|s| s.acetylcholine));

        HormoneTrend {
            cortisol: cortisol_trend,
            dopamine: dopamine_trend,
            acetylcholine: acetylcholine_trend,
        }
    }

    /// Calculate trend from values
    /// Note: values come from .iter().rev(), so first values are most recent
    fn calculate_trend<I>(values: I) -> Trend
    where
        I: Iterator<Item = f32>,
    {
        let vals: Vec<f32> = values.collect();
        if vals.len() < 2 {
            return Trend::Stable;
        }

        // first_half = recent values, second_half = older values
        let first_half: f32 = vals.iter().take(vals.len() / 2).sum::<f32>() / (vals.len() / 2) as f32;
        let second_half: f32 = vals.iter().skip(vals.len() / 2).sum::<f32>() / (vals.len() - vals.len() / 2) as f32;

        // If recent > old, trend is increasing
        let diff = first_half - second_half;
        if diff > 0.1 {
            Trend::Increasing
        } else if diff < -0.1 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }

    /// Get statistics
    pub fn stats(&self) -> EndocrineStats {
        EndocrineStats {
            current_mood: self.state.mood().to_string(),
            cortisol: self.state.cortisol,
            dopamine: self.state.dopamine,
            acetylcholine: self.state.acetylcholine,
            arousal: self.state.arousal(),
            valence: self.state.valence(),
            focus_streak: self.focus_streak,
            cycle_count: self.cycle_count,
        }
    }
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
}

/// Hormone trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HormoneTrend {
    pub cortisol: Trend,
    pub dopamine: Trend,
    pub acetylcholine: Trend,
}

/// Statistics from endocrine system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndocrineStats {
    pub current_mood: String,
    pub cortisol: f32,
    pub dopamine: f32,
    pub acetylcholine: f32,
    pub arousal: f32,
    pub valence: f32,
    pub focus_streak: u32,
    pub cycle_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hormone_state_neutral() {
        let state = HormoneState::neutral();
        assert!(state.cortisol > 0.0 && state.cortisol < 1.0);
        assert!(state.dopamine > 0.0 && state.dopamine < 1.0);
        assert!(state.acetylcholine > 0.0 && state.acetylcholine < 1.0);
    }

    #[test]
    fn test_hormone_state_moods() {
        let stressed = HormoneState::stressed();
        assert_eq!(stressed.mood(), "Anxious"); // High arousal, negative valence

        let excited = HormoneState::excited();
        assert_eq!(excited.mood(), "Excited"); // High arousal, positive valence
    }

    #[test]
    fn test_endocrine_system_creation() {
        let system = EndocrineSystem::new(EndocrineConfig::default());
        assert_eq!(system.state().mood(), "Calm");
    }

    #[test]
    fn test_error_increases_cortisol() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());
        let initial_cortisol = system.state().cortisol;

        system.process_event(HormoneEvent::Error { severity: 0.8 });

        assert!(system.state().cortisol > initial_cortisol);
    }

    #[test]
    fn test_success_increases_dopamine() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());
        let initial_dopamine = system.state().dopamine;

        system.process_event(HormoneEvent::Success { magnitude: 0.7 });

        assert!(system.state().dopamine > initial_dopamine);
    }

    #[test]
    fn test_deep_focus_increases_acetylcholine() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());
        let initial_ach = system.state().acetylcholine;

        system.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });

        assert!(system.state().acetylcholine > initial_ach);
        assert_eq!(system.focus_streak, 10);
    }

    #[test]
    fn test_context_switch_disrupts_focus() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());

        // Build up focus
        system.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });
        let focus_ach = system.state().acetylcholine;

        // Context switch
        system.process_event(HormoneEvent::ContextSwitch);

        assert!(system.state().acetylcholine < focus_ach);
        assert_eq!(system.focus_streak, 0);
    }

    #[test]
    fn test_hormone_decay() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());

        // Spike cortisol
        system.process_event(HormoneEvent::Threat { intensity: 1.0 });
        let high_cortisol = system.state().cortisol;

        // Decay over many cycles
        for _ in 0..100 {
            system.decay_cycle();
        }

        // Should return towards baseline (0.3)
        assert!(system.state().cortisol < high_cortisol);
        assert!((system.state().cortisol - 0.3).abs() < 0.2);
    }

    #[test]
    fn test_reward_learning_signal() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());
        let initial_dopamine = system.state().dopamine;

        system.process_event(HormoneEvent::Reward { value: 0.9 });

        assert!(system.state().dopamine > initial_dopamine);
        // Reward should give bigger dopamine spike than success
        let dopamine_after_reward = system.state().dopamine;

        let mut system2 = EndocrineSystem::new(EndocrineConfig::default());
        system2.process_event(HormoneEvent::Success { magnitude: 0.9 });

        assert!(dopamine_after_reward >= system2.state().dopamine);
    }

    #[test]
    fn test_recovery_accelerates_baseline() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());

        // Get stressed
        system.process_event(HormoneEvent::Threat { intensity: 1.0 });
        let stressed_cortisol = system.state().cortisol;

        // Recovery event
        system.process_event(HormoneEvent::Recovery);

        // Should be much closer to baseline
        assert!(system.state().cortisol < stressed_cortisol);
    }

    #[test]
    fn test_arousal_and_valence() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());

        // Get excited (high dopamine, low cortisol)
        system.process_event(HormoneEvent::Success { magnitude: 0.9 });
        system.process_event(HormoneEvent::Reward { value: 0.9 });

        let arousal = system.state().arousal();
        let valence = system.state().valence();

        assert!(arousal > 0.5); // Activated
        assert!(valence > 0.0); // Positive
    }

    #[test]
    fn test_trend_detection() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());

        // Build history of increasing cortisol
        for _ in 0..20 {
            system.process_event(HormoneEvent::Error { severity: 0.3 });
            system.decay_cycle();
        }

        let trend = system.get_trend(20);
        assert_eq!(trend.cortisol, Trend::Increasing);
    }

    #[test]
    fn test_stats() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());
        system.process_event(HormoneEvent::DeepFocus { duration_cycles: 5 });

        let stats = system.stats();
        assert_eq!(stats.focus_streak, 5);
        assert!(stats.acetylcholine > 0.4);
    }

    #[test]
    fn test_flow_state_bonus() {
        let mut system = EndocrineSystem::new(EndocrineConfig::default());
        let initial_dopamine = system.state().dopamine;

        // Build focus streak to trigger flow state
        system.process_event(HormoneEvent::DeepFocus { duration_cycles: 6 });

        // Flow state should give dopamine bonus
        assert!(system.state().dopamine > initial_dopamine);
    }
}
