// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Soul Manager — Value Alignment, Dissonance, Coherence
//!
//! Bridges soul value alignment into cognitive loop proposals via the
//! [`CognitiveSubsystem`] trait at interval 43 (co-prime).
//!
//! ## Design
//!
//! The manager receives alignment data from `Soul::evaluate_alignment()` via
//! `update_alignment()`, called in the strategy phase. On its scheduled
//! interval, `process()` converts alignment state into [`SubsystemOutput`]
//! proposals:
//!
//! - **High alignment (>0.6)**: Boosts confidence and learning rate — aligned
//!   actions deserve commitment (Schwartz 1992).
//! - **Low alignment (<0.3)**: Drops confidence, boosts exploration — misaligned
//!   state should trigger search for better alternatives.
//! - **Very low alignment (<0.1)**: Vetoes action — Festinger (1957) cognitive
//!   dissonance theory: acting against core values triggers avoidance.
//! - **Sustained misalignment**: Arousal increase — cognitive dissonance as
//!   somatic stress signal.
//!
//! ## Telemetry
//!
//! [`SoulTelemetry`] reports alignment EMA, coherence, misalignment streak,
//! and the most misaligned value for dashboard display.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};

// ── Constants ────────────────────────────────────────────────────────────────

/// Co-prime scheduling interval (cycles).
/// Science: Co-prime with other managers (7, 11, 13, 19, 37, 41, 47, 53, 67).
pub const SOUL_INTERVAL: u32 = 43;

/// Alignment above this → boost confidence and LR.
/// Basis: Schwartz (1992) — value congruence strengthens behavioral commitment.
pub const ALIGNMENT_BOOST_THRESHOLD: f32 = 0.6;

/// Alignment below this → drop confidence, boost exploration.
/// Basis: Festinger (1957) — dissonance drives attitude/behavior change.
pub const ALIGNMENT_CONCERN_THRESHOLD: f32 = 0.3;

/// Alignment below this → veto action entirely.
/// Basis: Extreme dissonance triggers avoidance behavior (Festinger 1957).
pub const ALIGNMENT_VETO_THRESHOLD: f32 = 0.1;

/// EMA smoothing factor for alignment tracking.
/// Slow EMA (0.1) prevents single-cycle noise from dominating.
pub const ALIGNMENT_EMA_ALPHA: f32 = 0.1;

/// Misalignment streak cycles before arousal escalation.
const DISSONANCE_ONSET_CYCLES: u32 = 5;

/// Maximum arousal boost from sustained dissonance.
const DISSONANCE_AROUSAL_MAX: f32 = 0.15;

// ── Telemetry ────────────────────────────────────────────────────────────────

/// Telemetry snapshot for dashboard/logging.
#[derive(Debug, Clone, Default)]
pub struct SoulTelemetry {
    /// Smoothed alignment EMA [0, 1].
    pub alignment_ema: f32,
    /// Soul coherence from last update.
    pub coherence: f32,
    /// Consecutive cycles below concern threshold.
    pub misalignment_streak: u32,
    /// Total experiences integrated by the Soul.
    pub experiences_integrated: u64,
    /// Name of the most misaligned value (if any).
    pub most_misaligned: Option<String>,
    /// Whether the manager has received data this cycle.
    pub has_update: bool,
}

// ── Manager ──────────────────────────────────────────────────────────────────

/// Soul Manager — value alignment → confidence, exploration, LR modulation.
///
/// Implements [`CognitiveSubsystem`] at interval 43 (co-prime).
/// Receives alignment data from `Soul::evaluate_alignment()` and proposes
/// cognitive loop adjustments based on value congruence.
pub struct SoulManager {
    /// Exponential moving average of alignment scores.
    alignment_ema: f32,
    /// Consecutive cycles with alignment below concern threshold.
    misalignment_streak: u32,
    /// Last raw alignment score received.
    last_alignment: f32,
    /// Last soul coherence value.
    last_coherence: f32,
    /// Last experiences_integrated count.
    last_experiences: u64,
    /// Name of the most misaligned value.
    most_misaligned: Option<String>,
    /// Whether `update_alignment` was called since last `process()`.
    has_update: bool,
}

impl SoulManager {
    /// Create a new SoulManager with neutral defaults.
    pub fn new() -> Self {
        Self {
            alignment_ema: 0.5,
            misalignment_streak: 0,
            last_alignment: 0.5,
            last_coherence: 0.0,
            last_experiences: 0,
            most_misaligned: None,
            has_update: false,
        }
    }

    /// Feed alignment data from the strategy phase.
    ///
    /// Called after `soul.evaluate_alignment()` with the results.
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
        self.most_misaligned = most_misaligned.map(|(name, _)| name);
        self.has_update = true;

        // Update EMA
        self.alignment_ema = self.alignment_ema * (1.0 - ALIGNMENT_EMA_ALPHA)
            + overall_alignment * ALIGNMENT_EMA_ALPHA;

        // Track misalignment streak
        if overall_alignment < ALIGNMENT_CONCERN_THRESHOLD {
            self.misalignment_streak = self.misalignment_streak.saturating_add(1);
        } else {
            self.misalignment_streak = 0;
        }
    }

    /// Current alignment EMA.
    pub fn alignment_ema(&self) -> f32 {
        self.alignment_ema
    }

    /// Current misalignment streak.
    pub fn misalignment_streak(&self) -> u32 {
        self.misalignment_streak
    }

    /// Telemetry snapshot for dashboard reporting.
    pub fn telemetry(&self) -> SoulTelemetry {
        SoulTelemetry {
            alignment_ema: self.alignment_ema,
            coherence: self.last_coherence,
            misalignment_streak: self.misalignment_streak,
            experiences_integrated: self.last_experiences,
            most_misaligned: self.most_misaligned.clone(),
            has_update: self.has_update,
        }
    }
}

impl Default for SoulManager {
    fn default() -> Self {
        Self::new()
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

        // Only produce proposals if we have alignment data
        if !self.has_update {
            return output;
        }

        let alignment = self.alignment_ema;

        // ── High alignment → boost confidence and learning ──────────────
        if alignment > ALIGNMENT_BOOST_THRESHOLD {
            let strength =
                (alignment - ALIGNMENT_BOOST_THRESHOLD) / (1.0 - ALIGNMENT_BOOST_THRESHOLD);
            // Boost confidence — aligned actions deserve commitment
            output.confidence_delta += strength as f64 * 0.05;
            // Boost learning rate — value-congruent experiences are salient
            output.lr_modulation = 1.0 + strength as f64 * 0.1;
        }

        // ── Low alignment → drop confidence, explore alternatives ───────
        if alignment < ALIGNMENT_CONCERN_THRESHOLD {
            let severity = (ALIGNMENT_CONCERN_THRESHOLD - alignment) / ALIGNMENT_CONCERN_THRESHOLD;
            // Drop confidence — misaligned actions shouldn't be committed to
            output.confidence_delta -= severity as f64 * 0.08;
            // Boost exploration — seek value-aligned alternatives
            output.exploration_delta += severity as f64 * 0.1;
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        // ── Very low alignment → veto action ────────────────────────────
        if alignment < ALIGNMENT_VETO_THRESHOLD {
            output.flags |= output_flags::VETO_ACTION;
        }

        // ── Sustained misalignment → cognitive dissonance arousal ────────
        if self.misalignment_streak > DISSONANCE_ONSET_CYCLES {
            let excess = (self.misalignment_streak - DISSONANCE_ONSET_CYCLES) as f32;
            let arousal_boost = (excess * 0.02).min(DISSONANCE_AROUSAL_MAX);
            output.arousal_delta += arousal_boost;
            // Negative valence from dissonance
            output.valence_delta -= arousal_boost * 0.5;
        }

        // Clear update flag after processing
        self.has_update = false;

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Format: [alignment_ema:f32][last_alignment:f32][last_coherence:f32]
        //         [misalignment_streak:u32][last_experiences:u64][has_update:u8]
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&self.alignment_ema.to_le_bytes());
        data.extend_from_slice(&self.last_alignment.to_le_bytes());
        data.extend_from_slice(&self.last_coherence.to_le_bytes());
        data.extend_from_slice(&self.misalignment_streak.to_le_bytes());
        data.extend_from_slice(&self.last_experiences.to_le_bytes());
        data.push(self.has_update as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 21 {
            return Err(format!(
                "SoulManager checkpoint too short: {} < 21",
                data.len()
            ));
        }
        self.alignment_ema = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "SoulManager: corrupt checkpoint bytes [0..4]".to_string())?,
        );
        self.last_alignment = f32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "SoulManager: corrupt checkpoint bytes [4..8]".to_string())?,
        );
        self.last_coherence = f32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "SoulManager: corrupt checkpoint bytes [8..12]".to_string())?,
        );
        self.misalignment_streak = u32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| "SoulManager: corrupt checkpoint bytes [12..16]".to_string())?,
        );
        self.last_experiences = u64::from_le_bytes(
            data[16..24]
                .try_into()
                .map_err(|_| "SoulManager: corrupt checkpoint bytes [16..24]".to_string())?,
        );
        if data.len() > 24 {
            self.has_update = data[24] != 0;
        }
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
        assert!((sm.alignment_ema - 0.5).abs() < 1e-6);
        assert_eq!(sm.misalignment_streak, 0);
        assert!(!sm.has_update);
    }

    #[test]
    fn test_name_and_interval() {
        let sm = SoulManager::new();
        assert_eq!(sm.name(), "soul_manager");
        assert_eq!(sm.interval(), 43);
    }

    #[test]
    fn test_no_output_without_update() {
        let mut sm = SoulManager::new();
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);
        assert!(output.is_neutral(), "Should produce neutral without update");
    }

    #[test]
    fn test_high_alignment_boosts() {
        let mut sm = SoulManager::new();
        // Feed high alignment repeatedly to push EMA above threshold
        for _ in 0..30 {
            sm.update_alignment(0.9, None, 0.8, 10);
        }
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);
        assert!(
            output.confidence_delta > 0.0,
            "High alignment should boost confidence: {}",
            output.confidence_delta
        );
        assert!(
            output.lr_modulation > 1.0,
            "High alignment should boost LR: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_low_alignment_explores() {
        let mut sm = SoulManager::new();
        // Feed low alignment repeatedly
        for _ in 0..30 {
            sm.update_alignment(0.1, Some(("honesty".into(), 0.05)), 0.3, 5);
        }
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
            output.flags & output_flags::REQUEST_EXPLORATION != 0,
            "Should request exploration"
        );
    }

    #[test]
    fn test_very_low_alignment_vetoes() {
        let mut sm = SoulManager::new();
        // Push EMA below veto threshold
        for _ in 0..50 {
            sm.update_alignment(0.02, None, 0.1, 2);
        }
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);
        assert!(
            output.flags & output_flags::VETO_ACTION != 0,
            "Very low alignment should veto action"
        );
    }

    #[test]
    fn test_sustained_misalignment_arousal() {
        let mut sm = SoulManager::new();
        // Feed misaligned values for many cycles
        for _ in 0..20 {
            sm.update_alignment(0.15, None, 0.2, 3);
        }
        assert!(sm.misalignment_streak > DISSONANCE_ONSET_CYCLES);
        let snapshot = CycleSnapshot::default();
        let output = sm.process(&snapshot);
        assert!(
            output.arousal_delta > 0.0,
            "Sustained misalignment should increase arousal: {}",
            output.arousal_delta
        );
        assert!(
            output.valence_delta < 0.0,
            "Dissonance should be aversive: {}",
            output.valence_delta
        );
    }

    #[test]
    fn test_misalignment_streak_resets() {
        let mut sm = SoulManager::new();
        for _ in 0..10 {
            sm.update_alignment(0.1, None, 0.2, 1);
        }
        assert!(sm.misalignment_streak > 0);
        // Feed high alignment
        sm.update_alignment(0.8, None, 0.9, 2);
        assert_eq!(sm.misalignment_streak, 0);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut sm = SoulManager::new();
        // Initial EMA = 0.5
        sm.update_alignment(1.0, None, 1.0, 1);
        // EMA should move toward 1.0 but not reach it instantly
        assert!(sm.alignment_ema > 0.5);
        assert!(sm.alignment_ema < 1.0);
        // After many updates at 1.0, should approach 1.0
        for _ in 0..100 {
            sm.update_alignment(1.0, None, 1.0, 1);
        }
        assert!(
            (sm.alignment_ema - 1.0).abs() < 0.01,
            "EMA should converge: {}",
            sm.alignment_ema
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut sm = SoulManager::new();
        sm.update_alignment(0.75, Some(("justice".into(), 0.3)), 0.6, 42);

        let checkpoint = sm.checkpoint();
        let mut sm2 = SoulManager::new();
        sm2.restore(&checkpoint).unwrap();

        assert!((sm2.alignment_ema - sm.alignment_ema).abs() < 1e-6);
        assert!((sm2.last_alignment - sm.last_alignment).abs() < 1e-6);
        assert!((sm2.last_coherence - sm.last_coherence).abs() < 1e-6);
        assert_eq!(sm2.misalignment_streak, sm.misalignment_streak);
        assert_eq!(sm2.last_experiences, sm.last_experiences);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut sm = SoulManager::new();
        let result = sm.restore(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_telemetry() {
        let mut sm = SoulManager::new();
        sm.update_alignment(0.6, Some(("courage".into(), 0.4)), 0.7, 15);
        let telem = sm.telemetry();
        assert!(telem.has_update);
        assert!((telem.coherence - 0.7).abs() < 1e-6);
        assert_eq!(telem.experiences_integrated, 15);
        assert_eq!(telem.most_misaligned, Some("courage".into()));
    }
}
