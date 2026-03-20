//! # Soul Manager — Value Alignment ↔ Consciousness Bridge
//!
//! Implements [`CognitiveSubsystem`] to transform soul value alignment into
//! cognitive proposals: confidence modulation, exploration gating, and learning
//! rate adjustment based on the Eight Harmonies core values.
//!
//! ## Design
//!
//! The soul alignment score is injected via `update_alignment()` from
//! `cycle_strategy.rs` (which already computes `soul.evaluate_alignment()`).
//! The manager then processes this alignment signal into proposals at interval
//! 43 (co-prime with all existing manager intervals).
//!
//! ## Value Alignment → Cognition
//!
//! - High alignment (>0.6): confidence boost, reduced exploration (proceed)
//! - Low alignment (<0.3): confidence drop, exploration boost (reconsider)
//! - Very low alignment (<0.1): action veto flag (Schwartz 1992, value-congruence theory)
//! - Misalignment streak: escalating arousal (Festinger 1957, cognitive dissonance)
//!
//! ## Telemetry
//!
//! Reports `SoulTelemetry` with alignment EMA, coherence, dissonance count,
//! and the most misaligned value name for introspection.

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Co-prime scheduling interval (cycles).
/// 43 is prime and distinct from all existing intervals (7, 11, 13, 19, 37, 41, 47, 53, 67).
const SOUL_INTERVAL: u32 = 43;

/// Alignment above this boosts confidence.
/// Basis: Schwartz (1992) — value-congruent actions produce approach motivation.
const ALIGNMENT_BOOST_THRESHOLD: f32 = 0.6;

/// Alignment below this triggers concern.
/// Basis: Festinger (1957) — value-incongruent states produce cognitive dissonance.
const ALIGNMENT_CONCERN_THRESHOLD: f32 = 0.3;

/// Alignment below this triggers action veto.
/// Basis: Schwartz (2012) — severe value violation triggers behavioral inhibition.
const ALIGNMENT_VETO_THRESHOLD: f32 = 0.1;

/// EMA decay for alignment tracking (0.9 = ~10 cycle memory).
const ALIGNMENT_EMA_ALPHA: f32 = 0.1;

/// Confidence boost scale when aligned.
const ALIGNED_CONFIDENCE_BOOST: f64 = 0.05;

/// Confidence drop scale when misaligned.
const MISALIGNED_CONFIDENCE_DROP: f64 = -0.08;

/// Exploration boost when misaligned (seek better options).
const MISALIGNED_EXPLORATION_BOOST: f64 = 0.06;

/// Arousal increment per cycle of sustained misalignment (dissonance stress).
const DISSONANCE_AROUSAL_INCREMENT: f32 = 0.02;

/// Maximum dissonance-driven arousal contribution.
const DISSONANCE_AROUSAL_MAX: f32 = 0.15;

/// Cycles of sustained misalignment before dissonance arousal kicks in.
const DISSONANCE_ONSET_CYCLES: u32 = 3;

/// LR boost when highly aligned (value-congruent learning is more plastic).
const ALIGNED_LR_BOOST: f64 = 1.1;

/// LR dampening when misaligned (slow down to reconsider).
const MISALIGNED_LR_DAMPEN: f64 = 0.85;

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Soul telemetry for cycle metadata reporting.
#[derive(Debug, Clone, Default)]
pub struct SoulTelemetry {
    /// Exponential moving average of value alignment (0.0–1.0 typical).
    pub alignment_ema: f32,
    /// Soul coherence (essence↔values similarity).
    pub soul_coherence: f32,
    /// Consecutive cycles of misalignment (dissonance counter).
    pub dissonance_streak: u32,
    /// Total value experiences integrated.
    pub experiences_integrated: u64,
    /// Name of the most misaligned value (for introspection).
    pub most_misaligned_value: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SOUL MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Soul Manager — bridges value alignment into cognitive loop proposals.
///
/// Implements `CognitiveSubsystem` at interval 43 (co-prime).
pub struct SoulManager {
    /// EMA-smoothed alignment score.
    alignment_ema: f32,
    /// Consecutive cycles below concern threshold.
    misalignment_streak: u32,
    /// Last raw alignment score (injected from cycle_strategy).
    last_alignment: f32,
    /// Last soul coherence (injected from soul.stats()).
    last_coherence: f32,
    /// Last experiences_integrated count.
    last_experiences: u64,
    /// Most misaligned value name.
    most_misaligned: String,
    /// Whether alignment data has been updated this cycle.
    has_update: bool,
}

impl Default for SoulManager {
    fn default() -> Self {
        Self {
            alignment_ema: 0.5, // neutral start
            misalignment_streak: 0,
            last_alignment: 0.5,
            last_coherence: 0.5,
            last_experiences: 0,
            most_misaligned: String::new(),
            has_update: false,
        }
    }
}

impl SoulManager {
    /// Create a new SoulManager with neutral initial state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inject alignment data from cycle_strategy's soul.evaluate_alignment() call.
    ///
    /// Called each cycle after alignment is computed (before process() runs).
    pub fn update_alignment(
        &mut self,
        overall_alignment: f32,
        most_misaligned: Option<(String, f32)>,
        coherence: f32,
        experiences_integrated: u64,
    ) {
        self.last_alignment = overall_alignment;
        self.last_coherence = coherence;
        self.last_experiences = experiences_integrated;
        if let Some((name, _score)) = most_misaligned {
            self.most_misaligned = name;
        }
        self.has_update = true;
    }

    /// Get current telemetry snapshot.
    pub fn telemetry(&self) -> SoulTelemetry {
        SoulTelemetry {
            alignment_ema: self.alignment_ema,
            soul_coherence: self.last_coherence,
            dissonance_streak: self.misalignment_streak,
            experiences_integrated: self.last_experiences,
            most_misaligned_value: self.most_misaligned.clone(),
        }
    }

    /// Current alignment EMA.
    pub fn alignment_ema(&self) -> f32 {
        self.alignment_ema
    }

    /// Whether the system is in cognitive dissonance (sustained misalignment).
    pub fn in_dissonance(&self) -> bool {
        self.misalignment_streak >= DISSONANCE_ONSET_CYCLES
    }
}

impl CognitiveSubsystem for SoulManager {
    fn name(&self) -> &'static str {
        "soul_manager"
    }

    fn interval(&self) -> u32 {
        SOUL_INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        if !self.has_update {
            return output;
        }
        self.has_update = false;

        // ── 1. Update alignment EMA ────────────────────────────────────────
        self.alignment_ema = self.alignment_ema * (1.0 - ALIGNMENT_EMA_ALPHA)
            + self.last_alignment * ALIGNMENT_EMA_ALPHA;

        let alignment = self.alignment_ema;

        // ── 2. High alignment → approach motivation ────────────────────────
        if alignment > ALIGNMENT_BOOST_THRESHOLD {
            // Value-congruent: boost confidence, enhance learning
            let strength = (alignment - ALIGNMENT_BOOST_THRESHOLD)
                / (1.0 - ALIGNMENT_BOOST_THRESHOLD);
            output.confidence_delta = ALIGNED_CONFIDENCE_BOOST * strength as f64;
            output.lr_modulation = 1.0 + (ALIGNED_LR_BOOST - 1.0) * strength as f64;

            // Reset dissonance
            self.misalignment_streak = 0;
        }
        // ── 3. Low alignment → cognitive dissonance ────────────────────────
        else if alignment < ALIGNMENT_CONCERN_THRESHOLD {
            let severity = (ALIGNMENT_CONCERN_THRESHOLD - alignment)
                / ALIGNMENT_CONCERN_THRESHOLD;
            output.confidence_delta = MISALIGNED_CONFIDENCE_DROP * severity as f64;
            output.exploration_delta = MISALIGNED_EXPLORATION_BOOST * severity as f64;
            output.lr_modulation = 1.0 + (MISALIGNED_LR_DAMPEN - 1.0) * severity as f64;

            // Track dissonance streak
            self.misalignment_streak = self.misalignment_streak.saturating_add(1);

            // Dissonance arousal (Festinger 1957)
            if self.misalignment_streak >= DISSONANCE_ONSET_CYCLES {
                let arousal = (DISSONANCE_AROUSAL_INCREMENT
                    * (self.misalignment_streak - DISSONANCE_ONSET_CYCLES) as f32)
                    .min(DISSONANCE_AROUSAL_MAX);
                output.arousal_delta += arousal;
            }

            // ── 4. Severe misalignment → veto ──────────────────────────────
            if alignment < ALIGNMENT_VETO_THRESHOLD {
                output.flags |= output_flags::VETO_ACTION;
            }
        } else {
            // Neutral zone: no strong signal, decay dissonance
            self.misalignment_streak = self.misalignment_streak.saturating_sub(1);
        }

        // ── 5. Soul coherence modulation ───────────────────────────────────
        // Low coherence = fragmented values → request rest/consolidation
        if self.last_coherence < 0.3 {
            output.flags |= output_flags::REQUEST_CONSOLIDATION;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(24);
        data.extend_from_slice(&self.alignment_ema.to_le_bytes());
        data.extend_from_slice(&self.misalignment_streak.to_le_bytes());
        data.extend_from_slice(&self.last_alignment.to_le_bytes());
        data.extend_from_slice(&self.last_coherence.to_le_bytes());
        data.extend_from_slice(&self.last_experiences.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 24 {
            return Err(format!(
                "SoulManager checkpoint too short: {} < 24",
                data.len()
            ));
        }
        self.alignment_ema = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "SoulManager: corrupt bytes [0..4]")?,
        );
        self.misalignment_streak = u32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "SoulManager: corrupt bytes [4..8]")?,
        );
        self.last_alignment = f32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "SoulManager: corrupt bytes [8..12]")?,
        );
        self.last_coherence = f32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| "SoulManager: corrupt bytes [12..16]")?,
        );
        self.last_experiences = u64::from_le_bytes(
            data[16..24]
                .try_into()
                .map_err(|_| "SoulManager: corrupt bytes [16..24]")?,
        );
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    #[test]
    fn test_soul_manager_defaults() {
        let sm = SoulManager::new();
        assert_eq!(sm.name(), "soul_manager");
        assert_eq!(sm.interval(), 43);
        assert!((sm.alignment_ema - 0.5).abs() < 1e-6);
        assert!(!sm.in_dissonance());
    }

    #[test]
    fn test_neutral_without_update() {
        let mut sm = SoulManager::new();
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);
        // No update injected → neutral output
        assert!((output.confidence_delta).abs() < 1e-9);
        assert!((output.lr_modulation - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_high_alignment_boosts_confidence() {
        let mut sm = SoulManager::new();
        // Start with high alignment EMA
        sm.alignment_ema = 0.8;
        sm.update_alignment(0.9, None, 0.7, 100);

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.confidence_delta > 0.0,
            "High alignment should boost confidence: {}",
            output.confidence_delta
        );
        assert!(
            output.lr_modulation > 1.0,
            "High alignment should boost learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_low_alignment_triggers_concern() {
        let mut sm = SoulManager::new();
        sm.alignment_ema = 0.1;
        sm.update_alignment(0.1, Some(("wisdom".into(), -0.5)), 0.5, 50);

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.confidence_delta < 0.0,
            "Low alignment should drop confidence: {}",
            output.confidence_delta
        );
        assert!(
            output.exploration_delta > 0.0,
            "Low alignment should boost exploration: {}",
            output.exploration_delta
        );
        assert!(
            output.lr_modulation < 1.0,
            "Low alignment should dampen learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_very_low_alignment_vetoes_action() {
        let mut sm = SoulManager::new();
        sm.alignment_ema = 0.05;
        sm.update_alignment(0.02, None, 0.5, 10);

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.flags & output_flags::VETO_ACTION != 0,
            "Severe misalignment should veto action"
        );
    }

    #[test]
    fn test_dissonance_streak_builds_arousal() {
        let mut sm = SoulManager::new();
        sm.alignment_ema = 0.15;
        let snapshot = CycleSnapshot::default();

        // Build up dissonance streak
        for i in 0..8 {
            sm.update_alignment(0.1, None, 0.5, i);
            let output = sm.process(&snapshot);

            if i >= DISSONANCE_ONSET_CYCLES as u64 {
                assert!(
                    output.arousal_delta > 0.0,
                    "Sustained misalignment should increase arousal at cycle {}: {}",
                    i,
                    output.arousal_delta
                );
            }
        }
        assert!(sm.in_dissonance());
    }

    #[test]
    fn test_dissonance_resets_on_alignment() {
        let mut sm = SoulManager::new();
        sm.alignment_ema = 0.15;
        sm.misalignment_streak = 10;
        assert!(sm.in_dissonance());

        // High alignment should reset
        sm.alignment_ema = 0.75;
        sm.update_alignment(0.8, None, 0.7, 100);
        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        assert_eq!(sm.misalignment_streak, 0);
        assert!(!sm.in_dissonance());
    }

    #[test]
    fn test_low_coherence_requests_consolidation() {
        let mut sm = SoulManager::new();
        sm.update_alignment(0.5, None, 0.2, 50); // low coherence

        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);

        assert!(
            output.flags & output_flags::REQUEST_CONSOLIDATION != 0,
            "Low soul coherence should request consolidation"
        );
    }

    #[test]
    fn test_alignment_ema_smoothing() {
        let mut sm = SoulManager::new();
        sm.alignment_ema = 0.5;

        // Sudden high alignment shouldn't immediately jump
        sm.update_alignment(1.0, None, 0.5, 1);
        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        // EMA: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        assert!(
            (sm.alignment_ema - 0.55).abs() < 0.01,
            "EMA should smooth: {}",
            sm.alignment_ema
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let sm = SoulManager {
            alignment_ema: 0.73,
            misalignment_streak: 5,
            last_alignment: 0.65,
            last_coherence: 0.81,
            last_experiences: 42,
            most_misaligned: "play".into(),
            has_update: false,
        };

        let checkpoint = sm.checkpoint();
        let mut sm2 = SoulManager::new();
        sm2.restore(&checkpoint).unwrap();

        assert!((sm2.alignment_ema - 0.73).abs() < 1e-6);
        assert_eq!(sm2.misalignment_streak, 5);
        assert!((sm2.last_alignment - 0.65).abs() < 1e-6);
        assert!((sm2.last_coherence - 0.81).abs() < 1e-6);
        assert_eq!(sm2.last_experiences, 42);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut sm = SoulManager::new();
        assert!(sm.restore(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_telemetry() {
        let mut sm = SoulManager::new();
        sm.update_alignment(0.7, Some(("stillness".into(), 0.1)), 0.85, 200);
        let snapshot = CycleSnapshot::default();
        sm.process(&snapshot);

        let tel = sm.telemetry();
        assert!(tel.alignment_ema > 0.0);
        assert!((tel.soul_coherence - 0.85).abs() < 1e-6);
        assert_eq!(tel.experiences_integrated, 200);
        assert_eq!(tel.most_misaligned_value, "stillness");
    }
}
