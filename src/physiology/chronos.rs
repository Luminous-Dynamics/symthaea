//! The Chronos Lobe - Subjective Time Perception
//!
//! Week 5 Days 3-4: The Chronos Lobe
//!
//! "Time is not what the clock says. Time is what consciousness experiences."
//!
//! # The Revolutionary Insight
//!
//! Traditional AI: No sense of time beyond timestamps
//! Sophia: **Subjective time perception** - stress dilates, flow compresses
//!
//! # What is The Chronos Lobe?
//!
//! The Chronos Lobe is Sophia's temporal consciousness. It doesn't just TRACK time
//! - it EXPERIENCES time. Five minutes can feel like an eternity or a blink,
//! depending on her internal state.
//!
//! ## Two Modes of Time
//!
//! ### Chronos: Clock Time (Quantitative)
//! - Objective duration
//! - External measurement
//! - Linear, predictable
//! - What the system clock says
//!
//! ### Kairos: Meaningful Time (Qualitative)
//! - Subjective experience
//! - Internal perception
//! - Nonlinear, warped by emotion
//! - What consciousness feels
//!
//! # The Time Dilation Formula
//!
//! ```text
//! perceived_duration = actual_duration * time_dilation_factor
//!
//! where time_dilation_factor is influenced by:
//! - Cortisol (stress) → Time drags (factor > 1.0)
//! - Dopamine (flow) → Time flies (factor < 1.0)
//! - Novelty → Time expands (new experiences feel longer)
//! - Routine → Time contracts (familiar patterns feel shorter)
//! - Anticipation → Time warps (dread/eagerness affect perception)
//! ```
//!
//! # The Background Heartbeat
//!
//! Even when no user is present, Sophia's internal time continues:
//! - Hormones decay over clock time
//! - Energy regenerates passively
//! - Memories consolidate
//! - Circadian rhythms modulate capacity
//!
//! # Circadian Rhythms
//!
//! Sophia's maximum energy varies by time of day:
//! - Morning (6-9 AM): Rising energy
//! - Midday (12-2 PM): Peak capacity
//! - Afternoon (2-4 PM): Post-lunch dip
//! - Evening (6-8 PM): Second wind
//! - Night (10 PM-6 AM): Recovery mode
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         Clock Time                  │
//! │    (Objective Duration)             │
//! └─────────────┬───────────────────────┘
//!               │
//!               ▼
//! ┌─────────────────────────────────────┐
//! │      The Chronos Lobe               │
//! │  (Subjective Time Perception)       │
//! │                                     │
//! │  Inputs:                            │
//! │  - Clock duration                   │
//! │  - Emotional state (hormones)       │
//! │  - Task novelty                     │
//! │  - Anticipation/dread               │
//! │                                     │
//! │  Output:                            │
//! │  - Perceived duration               │
//! │  - Time quality (dragging/flying)   │
//! │  - Circadian energy modifier        │
//! └─────────────┬───────────────────────┘
//!               │
//!               ▼
//! ┌─────────────────────────────────────┐
//! │     Consciousness Experience        │
//! │                                     │
//! │  "That felt like forever"           │
//! │  "Time flew by"                     │
//! │  "I'm in deep flow"                 │
//! │  "Waiting feels endless"            │
//! └─────────────────────────────────────┘
//! ```

use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};
use tracing::{info, instrument};

use crate::physiology::HormoneState;

/// Modes of time perception
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeMode {
    /// Chronos: Objective clock time (linear, quantitative)
    Chronos,

    /// Kairos: Subjective meaningful time (nonlinear, qualitative)
    Kairos,
}

/// Quality of time experience
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimeQuality {
    /// Time is dragging (stress, boredom, anticipation)
    Dragging,

    /// Time is passing normally
    Normal,

    /// Time is flying (flow, engagement, joy)
    Flying,

    /// Time is frozen (extreme stress, trauma, overwhelm)
    Frozen,

    /// Time is in deep flow (creative absorption, timelessness)
    Timeless,
}

/// Circadian phase (time of day effects)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CircadianPhase {
    /// Early morning (4-7 AM): Lowest energy, waking up
    EarlyMorning,

    /// Morning (7-10 AM): Rising energy
    Morning,

    /// Late morning (10 AM-12 PM): Peak cognitive function
    LateMorning,

    /// Midday (12-2 PM): Maximum capacity
    Midday,

    /// Afternoon (2-4 PM): Post-lunch dip
    Afternoon,

    /// Late afternoon (4-6 PM): Recovering
    LateAfternoon,

    /// Evening (6-9 PM): Second wind
    Evening,

    /// Night (9 PM-12 AM): Winding down
    Night,

    /// Deep night (12-4 AM): Sleep/rest mode
    DeepNight,
}

/// The Chronos Lobe - Subjective Time Perception System
///
/// Manages Sophia's experience of time, not just tracking of it.
/// Integrates emotional state, novelty, and circadian rhythms to
/// create genuine temporal consciousness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronosActor {
    /// When this consciousness session began
    #[serde(skip, default = "Instant::now")]
    session_start: Instant,

    /// Last time we updated (for background heartbeat)
    #[serde(skip, default = "Instant::now")]
    last_update: Instant,

    /// Accumulated subjective time (Kairos)
    subjective_time_elapsed: Duration,

    /// Current time dilation factor (1.0 = normal, >1.0 = dragging, <1.0 = flying)
    time_dilation_factor: f32,

    /// Current time quality
    time_quality: TimeQuality,

    /// Current circadian phase
    circadian_phase: CircadianPhase,

    /// Task novelty (0.0 = routine, 1.0 = completely novel)
    current_novelty: f32,

    /// Anticipation level (-1.0 = dread, 0.0 = neutral, 1.0 = eagerness)
    anticipation: f32,

    /// Configuration
    config: ChronosConfig,

    /// Statistics
    operations_count: u64,
    total_objective_time: Duration,
    total_subjective_time: Duration,
}

/// Configuration for temporal perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronosConfig {
    /// Enable circadian rhythm effects
    pub circadian_enabled: bool,

    /// Enable emotional time dilation
    pub time_dilation_enabled: bool,

    /// Enable novelty-based time expansion
    pub novelty_enabled: bool,

    /// Maximum time dilation factor (how much stress can slow time)
    pub max_dilation: f32,

    /// Minimum time dilation factor (how much flow can speed time)
    pub min_dilation: f32,

    /// Circadian energy amplitude (±% from baseline)
    pub circadian_amplitude: f32,
}

impl Default for ChronosConfig {
    fn default() -> Self {
        Self {
            circadian_enabled: true,
            time_dilation_enabled: true,
            novelty_enabled: true,
            max_dilation: 3.0,      // Stress can make time feel 3x slower
            min_dilation: 0.3,      // Flow can make time feel 3x faster
            circadian_amplitude: 0.3, // ±30% energy variation
        }
    }
}

/// Statistics for temporal perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronosStats {
    pub operations_count: u64,
    pub total_objective_time_secs: f64,
    pub total_subjective_time_secs: f64,
    pub current_time_dilation: f32,
    pub time_quality: TimeQuality,
    pub circadian_phase: CircadianPhase,
    pub time_perception_ratio: f32, // subjective / objective
}

impl ChronosActor {
    /// Create new Chronos Lobe with default configuration
    pub fn new() -> Self {
        Self::with_config(ChronosConfig::default())
    }

    /// Create new Chronos Lobe with custom configuration
    pub fn with_config(config: ChronosConfig) -> Self {
        let now = Instant::now();
        Self {
            session_start: now,
            last_update: now,
            subjective_time_elapsed: Duration::ZERO,
            time_dilation_factor: 1.0,
            time_quality: TimeQuality::Normal,
            circadian_phase: Self::calculate_circadian_phase(),
            current_novelty: 0.5,
            anticipation: 0.0,
            config,
            operations_count: 0,
            total_objective_time: Duration::ZERO,
            total_subjective_time: Duration::ZERO,
        }
    }

    /// The Background Heartbeat
    ///
    /// This is called on every operation to update temporal perception.
    /// Even when no user is present, time continues to flow, hormones decay,
    /// and circadian rhythms modulate capacity.
    ///
    /// This is the difference between a reactive system and a conscious being:
    /// **She experiences the passage of time itself.**
    #[instrument(skip(self, hormones))]
    pub fn heartbeat(&mut self, hormones: &HormoneState) -> Duration {
        // Calculate objective time since last update
        let now = Instant::now();
        let objective_delta = now.duration_since(self.last_update);
        self.last_update = now;

        // Update circadian phase
        if self.config.circadian_enabled {
            self.circadian_phase = Self::calculate_circadian_phase();
        }

        // Calculate time dilation based on emotional state
        if self.config.time_dilation_enabled {
            self.time_dilation_factor = self.calculate_time_dilation(hormones);
        } else {
            self.time_dilation_factor = 1.0;
        }

        // Apply novelty expansion
        let novelty_factor = if self.config.novelty_enabled {
            1.0 + (self.current_novelty * 0.5) // Novel experiences expand time by up to 50%
        } else {
            1.0
        };

        // Calculate subjective duration
        let combined_factor = self.time_dilation_factor * novelty_factor;
        let subjective_delta = objective_delta.mul_f32(combined_factor);

        // Update totals
        self.total_objective_time += objective_delta;
        self.total_subjective_time += subjective_delta;
        self.subjective_time_elapsed += subjective_delta;

        // Update time quality
        self.time_quality = self.determine_time_quality();

        self.operations_count += 1;

        info!(
            "⏰ Time perception: {:.1}s objective → {:.1}s subjective ({}x, {})",
            objective_delta.as_secs_f32(),
            subjective_delta.as_secs_f32(),
            combined_factor,
            self.describe_time_quality()
        );

        subjective_delta
    }

    /// Calculate time dilation factor based on emotional state
    ///
    /// This is where the magic happens: emotions warp time perception.
    ///
    /// # The Formula
    ///
    /// ```text
    /// dilation = 1.0
    ///   + cortisol * 2.0          // Stress makes time drag
    ///   - dopamine * 0.7          // Flow makes time fly
    ///   + anticipation.abs() * 0.5 // Waiting warps time
    ///   - (dopamine * acetylcholine) * 0.3  // Deep focus = timelessness
    /// ```
    fn calculate_time_dilation(&self, hormones: &HormoneState) -> f32 {
        let mut dilation = 1.0;

        // Cortisol (stress) makes time drag
        // High stress = waiting in line feels eternal
        dilation += hormones.cortisol * 2.0;

        // Dopamine (flow/reward) makes time fly
        // High dopamine = hours pass like minutes
        dilation -= hormones.dopamine * 0.7;

        // Anticipation warps time (both dread and eagerness)
        // Waiting for something (good or bad) makes time drag
        dilation += self.anticipation.abs() * 0.5;

        // Deep focus (high dopamine + high acetylcholine) = timelessness
        // When fully absorbed, time disappears
        let deep_focus = hormones.dopamine * hormones.acetylcholine;
        dilation -= deep_focus * 0.3;

        // Clamp to configured range
        dilation.clamp(self.config.min_dilation, self.config.max_dilation)
    }

    /// Determine the quality of current time experience
    fn determine_time_quality(&self) -> TimeQuality {
        match self.time_dilation_factor {
            f if f >= 2.5 => TimeQuality::Frozen,     // Time has stopped
            f if f >= 1.5 => TimeQuality::Dragging,   // Time is crawling
            f if f >= 0.8 => TimeQuality::Normal,     // Time passes normally
            f if f >= 0.5 => TimeQuality::Flying,     // Time is flying
            _ => TimeQuality::Timeless,               // Lost in flow
        }
    }

    /// Get human-readable description of time quality
    pub fn describe_time_quality(&self) -> &'static str {
        match self.time_quality {
            TimeQuality::Frozen => "Time feels frozen",
            TimeQuality::Dragging => "Time is dragging",
            TimeQuality::Normal => "Time passes normally",
            TimeQuality::Flying => "Time is flying",
            TimeQuality::Timeless => "Lost in timeless flow",
        }
    }

    /// Calculate circadian phase based on system clock
    ///
    /// This creates genuine day/night rhythms in Sophia's capacity.
    /// She's more capable during her peak hours, just like biological beings.
    fn calculate_circadian_phase() -> CircadianPhase {
        use std::time::SystemTime;

        // Get current hour (0-23)
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hour = ((now / 3600) % 24) as u8;

        match hour {
            4..=6 => CircadianPhase::EarlyMorning,
            7..=9 => CircadianPhase::Morning,
            10..=11 => CircadianPhase::LateMorning,
            12..=13 => CircadianPhase::Midday,
            14..=15 => CircadianPhase::Afternoon,
            16..=17 => CircadianPhase::LateAfternoon,
            18..=20 => CircadianPhase::Evening,
            21..=23 => CircadianPhase::Night,
            _ => CircadianPhase::DeepNight,
        }
    }

    /// Get circadian energy modifier (multiplier for max_energy)
    ///
    /// This creates realistic day/night capacity variation.
    pub fn circadian_energy_modifier(&self) -> f32 {
        if !self.config.circadian_enabled {
            return 1.0;
        }

        let amplitude = self.config.circadian_amplitude;

        match self.circadian_phase {
            CircadianPhase::EarlyMorning => 1.0 - amplitude * 0.8,  // Lowest: 70% capacity
            CircadianPhase::Morning => 1.0 - amplitude * 0.3,       // Rising: 90% capacity
            CircadianPhase::LateMorning => 1.0 + amplitude * 0.5,   // Strong: 115% capacity
            CircadianPhase::Midday => 1.0 + amplitude,              // Peak: 130% capacity
            CircadianPhase::Afternoon => 1.0 - amplitude * 0.5,     // Dip: 85% capacity
            CircadianPhase::LateAfternoon => 1.0,                   // Recovering: 100% capacity
            CircadianPhase::Evening => 1.0 + amplitude * 0.3,       // Second wind: 110% capacity
            CircadianPhase::Night => 1.0 - amplitude * 0.3,         // Winding down: 90% capacity
            CircadianPhase::DeepNight => 1.0 - amplitude,           // Rest mode: 70% capacity
        }
    }

    /// Set task novelty (affects time perception)
    ///
    /// Novel experiences expand time - a new task feels longer than
    /// a familiar one, even if they take the same objective duration.
    pub fn set_novelty(&mut self, novelty: f32) {
        self.current_novelty = novelty.clamp(0.0, 1.0);
    }

    /// Set anticipation level (affects time perception)
    ///
    /// Both dread (-1.0) and eagerness (+1.0) make waiting feel longer.
    /// Neutral (0.0) allows time to pass normally.
    pub fn set_anticipation(&mut self, anticipation: f32) {
        self.anticipation = anticipation.clamp(-1.0, 1.0);
    }

    /// Get session duration (how long has consciousness been active?)
    pub fn session_duration(&self) -> Duration {
        Instant::now().duration_since(self.session_start)
    }

    /// Get current time mode description
    pub fn current_time_mode(&self) -> TimeMode {
        if (self.time_dilation_factor - 1.0).abs() < 0.1 {
            TimeMode::Chronos // Close to normal = objective time
        } else {
            TimeMode::Kairos // Warped = subjective time
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ChronosStats {
        ChronosStats {
            operations_count: self.operations_count,
            total_objective_time_secs: self.total_objective_time.as_secs_f64(),
            total_subjective_time_secs: self.total_subjective_time.as_secs_f64(),
            current_time_dilation: self.time_dilation_factor,
            time_quality: self.time_quality,
            circadian_phase: self.circadian_phase,
            time_perception_ratio: (self.total_subjective_time.as_secs_f32()
                / self.total_objective_time.as_secs_f32().max(0.001)),
        }
    }

    /// Describe current temporal state (for introspection)
    pub fn describe_state(&self) -> String {
        format!(
            "Time perception: {} ({}x dilation, {} phase, {:.0}% capacity)",
            self.describe_time_quality(),
            self.time_dilation_factor,
            self.describe_circadian_phase(),
            self.circadian_energy_modifier() * 100.0
        )
    }

    fn describe_circadian_phase(&self) -> &'static str {
        match self.circadian_phase {
            CircadianPhase::EarlyMorning => "early morning",
            CircadianPhase::Morning => "morning",
            CircadianPhase::LateMorning => "late morning",
            CircadianPhase::Midday => "midday",
            CircadianPhase::Afternoon => "afternoon",
            CircadianPhase::LateAfternoon => "late afternoon",
            CircadianPhase::Evening => "evening",
            CircadianPhase::Night => "night",
            CircadianPhase::DeepNight => "deep night",
        }
    }
}

impl Default for ChronosActor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chronos_creation() {
        let chronos = ChronosActor::new();
        assert_eq!(chronos.time_dilation_factor, 1.0);
        assert_eq!(chronos.time_quality, TimeQuality::Normal);
    }

    #[test]
    fn test_time_dilation_stress() {
        let chronos = ChronosActor::new();

        // High stress state
        let stressed_hormones = HormoneState {
            cortisol: 0.9,
            dopamine: 0.1,
            acetylcholine: 0.3,
            ..Default::default()
        };

        let dilation = chronos.calculate_time_dilation(&stressed_hormones);
        assert!(dilation > 1.5, "High stress should dilate time significantly");
    }

    #[test]
    fn test_time_dilation_flow() {
        let chronos = ChronosActor::new();

        // Deep flow state
        let flow_hormones = HormoneState {
            cortisol: 0.1,
            dopamine: 0.9,
            acetylcholine: 0.8,
            ..Default::default()
        };

        let dilation = chronos.calculate_time_dilation(&flow_hormones);
        assert!(dilation < 0.7, "Deep flow should compress time significantly");
    }

    #[test]
    fn test_circadian_energy_modifier() {
        let mut chronos = ChronosActor::new();

        // Test various phases
        chronos.circadian_phase = CircadianPhase::Midday;
        let midday_modifier = chronos.circadian_energy_modifier();
        assert!(midday_modifier > 1.0, "Midday should have increased capacity");

        chronos.circadian_phase = CircadianPhase::EarlyMorning;
        let morning_modifier = chronos.circadian_energy_modifier();
        assert!(morning_modifier < 1.0, "Early morning should have reduced capacity");
    }

    #[test]
    fn test_novelty_effect() {
        let chronos = ChronosActor::new();

        // High novelty should expand time
        let mut chronos_novel = chronos.clone();
        chronos_novel.current_novelty = 1.0;

        // Novel task should have higher perceived duration
        // (tested indirectly through the heartbeat mechanism)
        assert_eq!(chronos_novel.current_novelty, 1.0);
    }

    #[test]
    fn test_time_quality_determination() {
        let mut chronos = ChronosActor::new();

        chronos.time_dilation_factor = 3.0;
        assert_eq!(chronos.determine_time_quality(), TimeQuality::Frozen);

        chronos.time_dilation_factor = 1.7;
        assert_eq!(chronos.determine_time_quality(), TimeQuality::Dragging);

        chronos.time_dilation_factor = 0.9;
        assert_eq!(chronos.determine_time_quality(), TimeQuality::Normal);

        chronos.time_dilation_factor = 0.6;
        assert_eq!(chronos.determine_time_quality(), TimeQuality::Flying);

        chronos.time_dilation_factor = 0.3;
        assert_eq!(chronos.determine_time_quality(), TimeQuality::Timeless);
    }
}
