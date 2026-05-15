// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Vision Manager — Visual Surprise → Exploration & Attention
//!
//! Consolidates vision-related cognitive feedback into a single [`CognitiveSubsystem`]
//! that reads from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Visual Processing Modeled
//!
//! 1. **Visual surprise**: Compressed state divergence drives exploration (Itti & Koch 2001)
//! 2. **Prediction error salience**: High visual PE → attention boost (Friston 2010)
//! 3. **Habituation**: Sustained low surprise → threshold adaptation (Rankin et al. 2009)
//! 4. **Saccade urgency**: Extreme surprise → arousal spike (Munoz & Everling 2004)
//!
//! ## Design
//!
//! Does NOT replace inline VisionBridge code in `cycle_phase_perception.rs` (dual-write bridge).
//! Proposes changes via SubsystemOutput which the OutputCollector integrates.

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};

/// Vision Manager — consolidates visual surprise, habituation, and saccade urgency.
///
/// Implements `CognitiveSubsystem` at interval 17 (co-prime).
pub struct VisionManager {
    /// Rolling window of visual surprise magnitudes (capacity: 16)
    surprise_window: [f32; 16],
    surprise_idx: usize,
    surprise_count: usize,

    /// Consecutive cycles of low visual surprise
    low_surprise_streak: u32,

    /// Adaptive surprise threshold (habituation)
    surprise_threshold: f32,

    /// EMA of visual prediction error
    visual_pe_ema: f32,

    /// Imagination-reality surprise from the vision manifold (P6-A).
    /// Set externally each cycle by the cognitive loop from VisionBridge::imagination_surprise().
    imagination_surprise: f32,
}

impl Default for VisionManager {
    fn default() -> Self {
        Self {
            surprise_window: [0.0; 16],
            surprise_idx: 0,
            surprise_count: 0,
            low_surprise_streak: 0,
            surprise_threshold: 0.2,
            visual_pe_ema: 0.0,
            imagination_surprise: 0.0,
        }
    }
}

impl VisionManager {
    /// Co-prime tick interval (17 is prime, co-prime with 7,11,13,19,23,29,31,37,41,43,47,53,59,67,71).
    pub const INTERVAL: u32 = 17;

    /// Below this, visual input is considered "familiar" (Itti & Koch 2001)
    const LOW_SURPRISE_THRESHOLD: f32 = 0.05;
    /// Habituation adaptation rate (Rankin et al. 2009)
    const THRESHOLD_ADAPT_RATE: f32 = 0.015;
    /// EMA smoothing factor for visual prediction error
    const PE_EMA_ALPHA: f32 = 0.15;
    /// Saccade urgency arousal threshold — extreme visual surprise
    const SACCADE_AROUSAL_THRESHOLD: f32 = 0.6;
    /// Exploration scale from visual surprise
    const SURPRISE_EXPLORATION_SCALE: f64 = 0.3;
    /// Arousal boost from saccade-level surprise
    const SACCADE_AROUSAL_BOOST: f32 = 0.1;
    /// Confidence reduction from high visual uncertainty
    const UNCERTAINTY_CONFIDENCE_PENALTY: f64 = -0.02;

    /// Current EMA of visual prediction error.
    pub fn visual_pe_ema(&self) -> f32 {
        self.visual_pe_ema
    }

    /// Current adaptive surprise threshold (increases with habituation).
    pub fn surprise_threshold(&self) -> f32 {
        self.surprise_threshold
    }

    /// Consecutive low-surprise cycles (habituation streak).
    pub fn low_surprise_streak(&self) -> u32 {
        self.low_surprise_streak
    }

    /// Set imagination-reality surprise from the vision manifold (P6-A).
    ///
    /// Called by the cognitive loop each cycle after processing vision frames.
    /// This signal is blended into the visual surprise computation in `process()`.
    pub fn set_imagination_surprise(&mut self, surprise: f32) {
        self.imagination_surprise = surprise.clamp(0.0, 1.0);
    }

    /// Current imagination surprise.
    pub fn imagination_surprise(&self) -> f32 {
        self.imagination_surprise
    }

    fn mean_surprise(&self) -> f32 {
        if self.surprise_count == 0 {
            return 0.0;
        }
        let sum: f32 = self.surprise_window[..self.surprise_count].iter().sum();
        sum / self.surprise_count as f32
    }

    fn push_surprise(&mut self, surprise: f32) {
        self.surprise_window[self.surprise_idx] = surprise;
        self.surprise_idx = (self.surprise_idx + 1) % 16;
        if self.surprise_count < 16 {
            self.surprise_count += 1;
        }
    }
}

impl CognitiveSubsystem for VisionManager {
    fn name(&self) -> &'static str {
        "vision_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // Visual surprise: derive from compressed_state variance + prediction_error
        // Higher compressed state variance → more visual complexity
        let state_variance = {
            let mean: f32 = snapshot.compressed_state.iter().copied().sum::<f32>() / 256.0;
            let var: f32 = snapshot
                .compressed_state
                .iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f32>()
                / 256.0;
            var.sqrt().min(1.0)
        };

        // Combine compressed state variance with prediction error for visual surprise.
        // P6-A: Imagination surprise amplifies the signal — when reality violates
        // the mental model, surprise is boosted by up to 30%.
        let base_surprise = state_variance * 0.4 + snapshot.prediction_error * 0.6;
        let imagination_boost = self.imagination_surprise * 0.3;
        let visual_surprise = (base_surprise + imagination_boost).min(1.0);

        self.push_surprise(visual_surprise);

        // EMA update
        self.visual_pe_ema =
            self.visual_pe_ema * (1.0 - Self::PE_EMA_ALPHA) + visual_surprise * Self::PE_EMA_ALPHA;

        // ── 0. FEP-driven Gating ─────────────────────────────────────
        // Science: Friston (2010). High Free Energy (F) indicates a poor world model.
        if snapshot.vision_free_energy > 0.5 {
            output.flags |= output_flags::VETO_ACTION;
            output.confidence_delta -= 0.05;
            tracing::warn!(
                f = snapshot.vision_free_energy,
                "Vision VETO: High Variational Free Energy"
            );
        }

        if snapshot.vision_free_energy > 0.3 {
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        // REQUEST_GEODESIC: If free energy is high, request mental simulation
        // to find a coherent path through the manifold (Active Inference).
        if snapshot.vision_free_energy > 0.2 {
            output.flags |= output_flags::REQUEST_GEODESIC;
        }

        // High complexity without accuracy → overfitting or noise sensitivity.
        // Dampen learning rate to stabilize.
        if snapshot.vision_complexity > 0.7 && snapshot.vision_accuracy < 0.5 {
            output.lr_modulation *= 0.8;
        }

        // ── 1. Habituation dynamics ─────────────────────────────────
        if visual_surprise < Self::LOW_SURPRISE_THRESHOLD {
            self.low_surprise_streak = self.low_surprise_streak.saturating_add(1);
        } else {
            self.low_surprise_streak = 0;
        }

        // ── 2. Surprise-driven exploration ──────────────────────────
        let surprise_signal = (visual_surprise - self.surprise_threshold).clamp(-0.5, 0.5);

        if surprise_signal > 0.0 {
            output.exploration_delta += surprise_signal as f64 * Self::SURPRISE_EXPLORATION_SCALE;
            // Adapt threshold upward (habituation)
            self.surprise_threshold += Self::THRESHOLD_ADAPT_RATE * surprise_signal;
        } else {
            // Sensitization: threshold drifts down
            self.surprise_threshold += Self::THRESHOLD_ADAPT_RATE * surprise_signal * 0.5;
        }
        self.surprise_threshold = self.surprise_threshold.clamp(0.05, 0.8);

        // ── 3. Saccade urgency: extreme surprise → arousal spike ────
        if visual_surprise > Self::SACCADE_AROUSAL_THRESHOLD {
            output.arousal_delta += Self::SACCADE_AROUSAL_BOOST;
            output.flags |= output_flags::ESCALATE_URGENCY;
        }

        // ── 4. High sustained uncertainty → confidence penalty ──────
        let mean_surp = self.mean_surprise();
        if mean_surp > 0.4 {
            output.confidence_delta += Self::UNCERTAINTY_CONFIDENCE_PENALTY;
        }

        // ── 5. Request exploration on novel visual input ────────────
        if surprise_signal > 0.1 {
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&self.surprise_threshold.to_le_bytes());
        data.extend_from_slice(&self.visual_pe_ema.to_le_bytes());
        data.extend_from_slice(&self.low_surprise_streak.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 12 {
            return Err(format!(
                "VisionManager checkpoint too short: {} < 12",
                data.len()
            ));
        }
        self.surprise_threshold = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "VisionManager: corrupt bytes [0..4]".to_string())?,
        );
        self.visual_pe_ema = f32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "VisionManager: corrupt bytes [4..8]".to_string())?,
        );
        self.low_surprise_streak = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "VisionManager: corrupt bytes [8..12]".to_string())?,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with_state_and_pe(state_val: f32, pe: f32) -> CycleSnapshot {
        let mut snap = CycleSnapshot::default();
        snap.prediction_error = pe;
        // Fill compressed state with a known value for controlled variance
        for s in snap.compressed_state.iter_mut() {
            *s = state_val;
        }
        snap
    }

    #[test]
    fn test_defaults() {
        let vm = VisionManager::default();
        assert_eq!(vm.visual_pe_ema, 0.0);
        assert!((vm.surprise_threshold - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_name_and_interval() {
        let vm = VisionManager::default();
        assert_eq!(vm.name(), "vision_manager");
        assert_eq!(vm.interval(), 17);
    }

    #[test]
    fn test_high_surprise_drives_exploration() {
        let mut vm = VisionManager::default();
        let snap = snapshot_with_state_and_pe(0.0, 0.8);
        let output = vm.process(&snap);
        assert!(
            output.exploration_delta > 0.0,
            "High PE should drive exploration: {}",
            output.exploration_delta
        );
    }

    #[test]
    fn test_saccade_urgency_on_extreme_surprise() {
        let mut vm = VisionManager::default();
        // Need visual_surprise > 0.6 (SACCADE_AROUSAL_THRESHOLD).
        // visual_surprise = state_variance * 0.4 + pe * 0.6
        // With state_val=0.0 (zero variance) and pe=1.0: 0.0 + 0.6 = 0.6, not enough.
        // Use non-uniform state to add variance contribution.
        let mut snap = CycleSnapshot::default();
        snap.prediction_error = 1.0;
        // Create high variance: half zeros, half ones
        for (i, s) in snap.compressed_state.iter_mut().enumerate() {
            *s = if i % 2 == 0 { 0.0 } else { 1.0 };
        }
        let output = vm.process(&snap);
        assert!(
            output.arousal_delta > 0.0,
            "Extreme surprise should boost arousal"
        );
        assert!(output.has_flag(output_flags::ESCALATE_URGENCY));
    }

    #[test]
    fn test_threshold_adapts_on_repeated_surprise() {
        let mut vm = VisionManager::default();
        let initial = vm.surprise_threshold;
        let snap = snapshot_with_state_and_pe(0.0, 0.8);
        for _ in 0..20 {
            vm.process(&snap);
        }
        assert!(
            vm.surprise_threshold > initial,
            "Threshold should habituate upward"
        );
    }

    #[test]
    fn test_low_surprise_no_exploration() {
        let mut vm = VisionManager::default();
        let snap = snapshot_with_state_and_pe(0.5, 0.01);
        let output = vm.process(&snap);
        // With very low PE and uniform state (low variance), surprise is low
        // exploration_delta should be small or negative
        assert!(
            output.exploration_delta <= 0.01,
            "Low surprise should not drive exploration: {}",
            output.exploration_delta
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let vm = VisionManager {
            surprise_threshold: 0.45,
            visual_pe_ema: 0.32,
            low_surprise_streak: 7,
            ..Default::default()
        };
        let data = vm.checkpoint();
        let mut vm2 = VisionManager::default();
        vm2.restore(&data).unwrap();
        assert!((vm2.surprise_threshold - 0.45).abs() < 1e-6);
        assert!((vm2.visual_pe_ema - 0.32).abs() < 1e-6);
        assert_eq!(vm2.low_surprise_streak, 7);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut vm = VisionManager::default();
        assert!(vm.restore(&[0u8; 5]).is_err());
    }

    // ── Property Tests ────────────────────────────────────────────────────
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_vision_exploration_bounded(
            pe in 0.0f32..1.0,
            state_val in -1.0f32..1.0
        ) {
            let mut vm = VisionManager::default();
            let mut snap = CycleSnapshot::default();
            snap.prediction_error = pe;
            for s in snap.compressed_state.iter_mut() { *s = state_val; }
            let output = vm.process(&snap);
            prop_assert!(output.exploration_delta.abs() < 1.0,
                "exploration_delta out of bounds: {}", output.exploration_delta);
            prop_assert!(output.arousal_delta.abs() < 1.0,
                "arousal_delta out of bounds: {}", output.arousal_delta);
        }

        #[test]
        fn prop_vision_threshold_bounded(
            pe in 0.0f32..1.0
        ) {
            let mut vm = VisionManager::default();
            let snap = CycleSnapshot { prediction_error: pe, ..Default::default() };
            for _ in 0..100 {
                vm.process(&snap);
            }
            prop_assert!(vm.surprise_threshold() >= 0.05,
                "threshold below min: {}", vm.surprise_threshold());
            prop_assert!(vm.surprise_threshold() <= 0.8,
                "threshold above max: {}", vm.surprise_threshold());
        }

        #[test]
        fn prop_vision_pe_ema_bounded(
            pe in 0.0f32..1.0
        ) {
            let mut vm = VisionManager::default();
            let snap = CycleSnapshot { prediction_error: pe, ..Default::default() };
            for _ in 0..50 {
                vm.process(&snap);
            }
            prop_assert!(vm.visual_pe_ema() >= 0.0,
                "pe_ema below 0: {}", vm.visual_pe_ema());
            prop_assert!(vm.visual_pe_ema() <= 1.0,
                "pe_ema above 1: {}", vm.visual_pe_ema());
        }
    }
}
