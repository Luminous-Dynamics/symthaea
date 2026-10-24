// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Drive Manager — Curiosity, Flow, Boredom, Exploration
//!
//! Consolidates drive-related logic into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Drives Modeled
//!
//! 1. **Curiosity**: Prediction errors that are moderate trigger exploration (Berlyne 1960)
//! 2. **Boredom**: Sustained low error → exploration urge ramps up (Eastwood 2012)
//! 3. **Flow**: Optimal challenge-skill balance → suppress exploration, boost learning (Csikszentmihalyi 1990)
//! 4. **Exploration**: Surprise-driven threshold for novelty seeking (Friston 2010)
//!
//! ## Design
//!
//! The manager maintains internal state (rolling windows, counters) but does NOT mutate
//! any CognitiveLoopService fields directly. Instead, it proposes changes via
//! `SubsystemOutput` which the `OutputCollector` integrates via consensus averaging.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thresholds;
use crate::cognitive_loop::types::PredictionEvidenceCursor;

/// Drive Manager — consolidates curiosity, boredom, flow, and exploration drives.
///
/// Implements `CognitiveSubsystem` at interval 7 (co-prime).
pub struct DriveManager {
    // ── Curiosity state ───────────────────────────────────────────────
    /// Rolling window of recent prediction errors (capacity: 16)
    error_window: [f32; 16],
    /// Write index into error_window (ring buffer)
    error_idx: usize,
    /// Number of errors recorded (max 16)
    error_count: usize,

    // ── Boredom state ─────────────────────────────────────────────────
    /// Consecutive cycles with low prediction error
    low_error_streak: u32,
    /// Current boredom level [0, 1]
    boredom: f32,

    // ── Flow state ────────────────────────────────────────────────────
    /// Consecutive cycles meeting flow criteria
    flow_streak: u32,
    /// Whether currently in flow
    in_flow: bool,
    /// Flow intensity [0, 1]
    flow_intensity: f32,

    // ── Exploration state ─────────────────────────────────────────────
    /// Adaptive exploration threshold (surprise must exceed this)
    exploration_threshold: f32,
    /// Prevents duplicate, stale, or out-of-order PE samples from changing drive state.
    prediction_error_cursor: PredictionEvidenceCursor,
}

impl Default for DriveManager {
    fn default() -> Self {
        Self {
            error_window: [0.0; 16],
            error_idx: 0,
            error_count: 0,
            low_error_streak: 0,
            boredom: 0.0,
            flow_streak: 0,
            in_flow: false,
            flow_intensity: 0.0,
            exploration_threshold: 0.3,
            prediction_error_cursor: PredictionEvidenceCursor::default(),
        }
    }
}

impl DriveManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 7;

    /// Threshold below which prediction error is "low" (boredom territory).
    /// Basis: Eastwood et al. (2012) — boredom as failed engagement with environment.
    const LOW_ERROR_THRESHOLD: f32 = 0.05;

    /// Cycles of sustained low error before boredom kicks in.
    const BOREDOM_ONSET_CYCLES: u32 = 20;

    /// Maximum boredom level (capped to prevent runaway exploration).
    const MAX_BOREDOM: f32 = 0.8;

    /// Boredom decay per cycle when error is not low (EMA pull toward 0).
    const BOREDOM_DECAY: f32 = 0.95;

    /// Flow entry: error must be below this AND coherence above FLOW_COHERENCE_MIN.
    /// Basis: Csikszentmihalyi (1990) — challenge ≈ skill balance.
    const FLOW_ERROR_MAX: f32 = 0.15;

    /// Flow entry: coherence must be above this.
    const FLOW_COHERENCE_MIN: f32 = 0.6;

    /// Cycles of meeting flow criteria before entering flow.
    const FLOW_ONSET_CYCLES: u32 = 5;

    /// Flow intensity ramp rate per cycle (while in flow).
    const FLOW_RAMP: f32 = 0.05;

    /// Flow intensity decay rate per cycle (while not in flow).
    const FLOW_DECAY: f32 = 0.1;

    /// Exploration threshold adaptation rate.
    /// Basis: Friston (2010) — precision-weighted surprise modulates exploration.
    const THRESHOLD_ADAPT_RATE: f32 = 0.02;

    /// Current boredom level [0, MAX_BOREDOM].
    pub fn boredom(&self) -> f32 {
        self.boredom
    }

    /// Current flow intensity [0, 1].
    pub fn flow_intensity(&self) -> f32 {
        self.flow_intensity
    }

    /// Whether currently in flow state.
    pub fn in_flow(&self) -> bool {
        self.in_flow
    }

    /// Adaptive exploration threshold (surprise must exceed this).
    pub fn exploration_threshold(&self) -> f32 {
        self.exploration_threshold
    }

    /// Compute mean prediction error from the rolling window.
    fn mean_error(&self) -> f32 {
        if self.error_count == 0 {
            return 0.0;
        }
        let sum: f32 = self.error_window[..self.error_count].iter().sum();
        sum / self.error_count as f32
    }

    /// Push a new prediction error into the rolling window.
    fn push_error(&mut self, error: f32) {
        self.error_window[self.error_idx] = error;
        self.error_idx = (self.error_idx + 1) % 16;
        if self.error_count < 16 {
            self.error_count += 1;
        }
    }
}

impl CognitiveSubsystem for DriveManager {
    fn name(&self) -> &'static str {
        "drive_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Record prediction error ────────────────────────────────────
        let prediction_error_sample = self
            .prediction_error_cursor
            .admit_evidence(&snapshot.prediction_error_evidence(), snapshot.cycle_number);
        let prediction_error_valid = prediction_error_sample.is_some();
        let pe = prediction_error_sample.unwrap_or(0.0);
        if prediction_error_valid {
            self.push_error(pe);
        }
        let _mean_pe = self.mean_error();

        // ── 2. Boredom dynamics ───────────────────────────────────────────
        // Boredom responds only to genuine prediction comparisons; missing evidence
        // must neither count as a surprising failure nor as an easy low-error cycle.
        if prediction_error_valid {
            if pe < Self::LOW_ERROR_THRESHOLD {
                self.low_error_streak = self.low_error_streak.saturating_add(1);
                if self.low_error_streak > Self::BOREDOM_ONSET_CYCLES {
                    self.boredom =
                        (self.boredom + thresholds::DRIVE_BOREDOM_INCREMENT).min(Self::MAX_BOREDOM);
                }
            } else {
                self.low_error_streak = 0;
                self.boredom *= Self::BOREDOM_DECAY;
            }
        }

        // Boredom drives exploration
        if self.boredom > thresholds::DRIVE_BOREDOM_EXPLORATION_THRESHOLD {
            output.exploration_delta +=
                (self.boredom - thresholds::DRIVE_BOREDOM_EXPLORATION_THRESHOLD) as f64
                    * thresholds::DRIVE_BOREDOM_EXPLORATION_SCALE;
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        // ── 3. Flow detection ─────────────────────────────────────────────
        // Flow = sustained low error + high coherence (optimal challenge-skill)
        let meets_flow = pe < Self::FLOW_ERROR_MAX
            && snapshot.coherence > Self::FLOW_COHERENCE_MIN
            && snapshot.arousal > thresholds::DRIVE_FLOW_AROUSAL_MIN
            && snapshot.arousal < thresholds::DRIVE_FLOW_AROUSAL_MAX;

        if prediction_error_valid {
            if meets_flow {
                self.flow_streak = self.flow_streak.saturating_add(1);
                if self.flow_streak >= Self::FLOW_ONSET_CYCLES {
                    self.in_flow = true;
                    self.flow_intensity = (self.flow_intensity + Self::FLOW_RAMP).min(1.0);
                }
            } else {
                self.flow_streak = 0;
                if self.in_flow {
                    self.flow_intensity -= Self::FLOW_DECAY;
                    if self.flow_intensity <= 0.0 {
                        self.in_flow = false;
                        self.flow_intensity = 0.0;
                    }
                }
            }
        }

        // Flow suppresses exploration, boosts learning
        if self.in_flow {
            // In flow: dampen exploration (don't seek novelty when optimally engaged)
            output.exploration_delta -=
                self.flow_intensity as f64 * thresholds::DRIVE_FLOW_EXPLORATION_DAMPEN;
            // In flow: boost learning rate (peak plasticity window)
            output.lr_modulation =
                1.0 + self.flow_intensity as f64 * thresholds::DRIVE_FLOW_LR_BOOST;
            // In flow: stabilize confidence
            output.confidence_delta +=
                self.flow_intensity as f64 * thresholds::DRIVE_FLOW_CONFIDENCE_BOOST;
            // In flow: boredom resets
            self.boredom *= thresholds::DRIVE_FLOW_BOREDOM_RESET;
        }

        // ── 4. Curiosity / exploration dynamics ───────────────────────────
        // Moderate surprise → exploration. Too much surprise → retreat.
        let surprise_signal = if prediction_error_valid {
            (pe - self.exploration_threshold).clamp(-0.5, 0.5)
        } else {
            0.0
        };

        if prediction_error_valid {
            if surprise_signal > 0.0 {
                let exploration_boost =
                    surprise_signal as f64 * thresholds::DRIVE_SURPRISE_EXPLORATION_SCALE;
                output.exploration_delta += exploration_boost;
                output.arousal_delta += surprise_signal * thresholds::DRIVE_SURPRISE_AROUSAL_SCALE;
                self.exploration_threshold += Self::THRESHOLD_ADAPT_RATE * surprise_signal;
            } else {
                self.exploration_threshold += Self::THRESHOLD_ADAPT_RATE * surprise_signal * 0.5;
            }
            self.exploration_threshold = self.exploration_threshold.clamp(0.05, 0.8);
        }

        // ── 5. Emotional homeostasis contribution ─────────────────────────
        // Pull valence toward neutral when no strong drive signal
        if prediction_error_valid
            && !self.in_flow
            && self.boredom < 0.2
            && surprise_signal.abs() < 0.1
        {
            // Gentle pull toward neutral valence
            let valence_pull = -snapshot.valence * 0.02;
            output.valence_delta += valence_pull;
        }

        // ── 7. High somatic stress → suppress exploration ─────────────────
        if snapshot.somatic_stress > 0.5 {
            output.exploration_delta *= 0.5;
            // Stress dampens learning
            output.lr_modulation *= 1.0 - (snapshot.somatic_stress - 0.5) * 0.3;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Preserve the legacy 25-byte state prefix. PEVC v2 appends both cycle
        // watermarks so invalid newer observations still block delayed older PE
        // after restart while allowing a valid retry at that same cycle.
        let mut data = Vec::with_capacity(48);
        data.extend_from_slice(&self.boredom.to_le_bytes());
        data.extend_from_slice(&self.flow_intensity.to_le_bytes());
        data.extend_from_slice(&self.exploration_threshold.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes()); // reserved legacy slot
        data.push(self.in_flow as u8);
        data.extend_from_slice(&self.low_error_streak.to_le_bytes());
        data.extend_from_slice(&self.flow_streak.to_le_bytes());
        data.extend_from_slice(b"PEVC");
        data.push(2); // extension version
        let observed_cycle = self
            .prediction_error_cursor
            .last_observed_observation_cycle();
        data.push(u8::from(observed_cycle.is_some()));
        data.extend_from_slice(&observed_cycle.unwrap_or_default().to_le_bytes());
        let admitted_cycle = self
            .prediction_error_cursor
            .last_admitted_observation_cycle();
        data.push(u8::from(admitted_cycle.is_some()));
        data.extend_from_slice(&admitted_cycle.unwrap_or_default().to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() != 25 && data.len() != 48 {
            return Err(format!(
                "DriveManager checkpoint has unsupported length: {}",
                data.len()
            ));
        }

        // Validate the extension before mutating manager state. Legacy 25-byte
        // checkpoints contain no cursor, so restore an empty cursor rather than
        // retaining a watermark from this manager instance.
        let (restored_observed_cycle, restored_admitted_cycle) = if data.len() == 25 {
            (None, None)
        } else {
            if &data[25..29] != b"PEVC" {
                return Err("DriveManager: invalid PE cursor checkpoint magic".to_string());
            }
            if data[29] != 2 {
                return Err(format!(
                    "DriveManager: unsupported PE cursor checkpoint version {}",
                    data[29]
                ));
            }
            let has_observed = data[30];
            let observed_value = u64::from_le_bytes(
                data[31..39]
                    .try_into()
                    .map_err(|_| "DriveManager: corrupt observed-cycle bytes".to_string())?,
            );
            let has_admitted = data[39];
            let admitted_value = u64::from_le_bytes(
                data[40..48]
                    .try_into()
                    .map_err(|_| "DriveManager: corrupt admitted-cycle bytes".to_string())?,
            );
            if has_observed > 1 || has_admitted > 1 {
                return Err("DriveManager: invalid PE cursor presence flag".to_string());
            }
            let observed = match has_observed {
                1 => Some(observed_value),
                0 if observed_value == 0 => None,
                0 => {
                    return Err(
                        "DriveManager: absent observed cycle must encode zero".to_string()
                    );
                }
                _ => unreachable!("presence flag was checked above"),
            };
            let admitted = match has_admitted {
                1 => Some(admitted_value),
                0 if admitted_value == 0 => None,
                0 => {
                    return Err(
                        "DriveManager: absent admitted cycle must encode zero".to_string()
                    );
                }
                _ => unreachable!("presence flag was checked above"),
            };
            if admitted.is_some_and(|cycle| observed.is_none_or(|observed_cycle| cycle > observed_cycle)) {
                return Err(
                    "DriveManager: admitted PE cycle exceeds observed high-water mark".to_string(),
                );
            }
            (observed, admitted)
        };

        self.boredom = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "DriveManager: corrupt checkpoint bytes [0..4]".to_string())?,
        );
        self.flow_intensity = f32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "DriveManager: corrupt checkpoint bytes [4..8]".to_string())?,
        );
        self.exploration_threshold = f32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "DriveManager: corrupt checkpoint bytes [8..12]".to_string())?,
        );
        // bytes [12..16] reserved (was exploration_urge) — skip
        self.in_flow = data[16] != 0;
        self.low_error_streak = u32::from_le_bytes(
            data[17..21]
                .try_into()
                .map_err(|_| "DriveManager: corrupt checkpoint bytes [17..21]".to_string())?,
        );
        self.flow_streak = u32::from_le_bytes(
            data[21..25]
                .try_into()
                .map_err(|_| "DriveManager: corrupt checkpoint bytes [21..25]".to_string())?,
        );
        self.prediction_error_cursor
            .restore_cycles(restored_observed_cycle, restored_admitted_cycle);
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

    fn snapshot_with(pe: f64, coherence: f64, arousal: f32, valence: f32) -> CycleSnapshot {
        CycleSnapshot {
            prediction_error: pe as f32,
            prediction_error_valid: 1,
            coherence: coherence as f32,
            arousal,
            valence,
            ..Default::default()
        }
    }

    fn process_cycles(
        manager: &mut DriveManager,
        snapshot: &CycleSnapshot,
        start_cycle: u64,
        count: usize,
    ) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;
        for cycle in start_cycle..start_cycle + count as u64 {
            let mut current = *snapshot;
            current.cycle_number = cycle;
            output = manager.process(&current);
        }
        output
    }

    #[test]
    fn test_drive_manager_defaults() {
        let dm = DriveManager::default();
        assert_eq!(dm.boredom, 0.0);
        assert!(!dm.in_flow);
        assert_eq!(dm.flow_intensity, 0.0);
    }

    #[test]
    fn test_name_and_interval() {
        let dm = DriveManager::default();
        assert_eq!(dm.name(), "drive_manager");
        assert_eq!(dm.interval(), 7);
    }

    #[test]
    fn test_boredom_accumulates_on_low_error() {
        let mut dm = DriveManager::default();
        let snapshot = snapshot_with(0.01, 0.5, 0.5, 0.0);

        // Run enough cycles to trigger boredom
        process_cycles(&mut dm, &snapshot, 0, 30);

        assert!(
            dm.boredom > 0.0,
            "Boredom should accumulate: {}",
            dm.boredom
        );
    }

    #[test]
    fn test_boredom_drives_exploration() {
        let mut dm = DriveManager::default();
        let snapshot = snapshot_with(0.01, 0.5, 0.5, 0.0);

        // Accumulate boredom
        process_cycles(&mut dm, &snapshot, 0, 50);

        let output = process_cycles(&mut dm, &snapshot, 50, 1);
        assert!(
            output.exploration_delta > 0.0,
            "Boredom should drive exploration: {}",
            output.exploration_delta
        );
        assert!(
            output.flags & output_flags::REQUEST_EXPLORATION != 0,
            "Should request exploration flag"
        );
    }

    #[test]
    fn test_boredom_decays_on_surprise() {
        let mut dm = DriveManager::default();

        // Build up boredom
        let boring = snapshot_with(0.01, 0.5, 0.5, 0.0);
        process_cycles(&mut dm, &boring, 0, 50);
        let boredom_peak = dm.boredom;
        assert!(boredom_peak > 0.0);

        // Hit with surprise
        let surprising = snapshot_with(0.5, 0.5, 0.5, 0.0);
        process_cycles(&mut dm, &surprising, 50, 10);

        assert!(
            dm.boredom < boredom_peak,
            "Boredom should decay on surprise: {} < {}",
            dm.boredom,
            boredom_peak
        );
    }

    #[test]
    fn test_flow_detection() {
        let mut dm = DriveManager::default();
        // Flow requires: low error + high coherence + moderate arousal
        let flow_snapshot = snapshot_with(0.05, 0.8, 0.5, 0.3);

        // Need FLOW_ONSET_CYCLES (5) consecutive cycles
        process_cycles(&mut dm, &flow_snapshot, 0, 10);

        assert!(dm.in_flow, "Should enter flow state");
        assert!(dm.flow_intensity > 0.0, "Flow intensity should be positive");
    }

    #[test]
    fn test_flow_suppresses_exploration() {
        let mut dm = DriveManager::default();
        let flow_snapshot = snapshot_with(0.05, 0.8, 0.5, 0.3);

        // Enter flow
        process_cycles(&mut dm, &flow_snapshot, 0, 10);
        assert!(dm.in_flow);

        let output = process_cycles(&mut dm, &flow_snapshot, 10, 1);
        assert!(
            output.exploration_delta < 0.0,
            "Flow should suppress exploration: {}",
            output.exploration_delta
        );
        assert!(
            output.lr_modulation > 1.0,
            "Flow should boost learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_flow_exits_on_disruption() {
        let mut dm = DriveManager::default();
        let flow_snapshot = snapshot_with(0.05, 0.8, 0.5, 0.3);

        // Enter flow
        process_cycles(&mut dm, &flow_snapshot, 0, 10);
        assert!(dm.in_flow);

        // Disrupt with high error
        let disruption = snapshot_with(0.5, 0.3, 0.8, -0.2);
        process_cycles(&mut dm, &disruption, 10, 20);

        assert!(!dm.in_flow, "Should exit flow on disruption");
        assert_eq!(dm.flow_intensity, 0.0);
    }

    #[test]
    fn test_surprise_drives_exploration() {
        let mut dm = DriveManager::default();
        let surprising = snapshot_with(0.8, 0.5, 0.5, 0.0);

        let output = dm.process(&surprising);
        assert!(
            output.exploration_delta > 0.0,
            "Surprise should drive exploration: {}",
            output.exploration_delta
        );
        assert!(
            output.arousal_delta > 0.0,
            "Surprise should increase arousal: {}",
            output.arousal_delta
        );
    }

    #[test]
    fn test_exploration_threshold_adapts() {
        let mut dm = DriveManager::default();
        let initial_threshold = dm.exploration_threshold;

        // Repeated surprising input → threshold should rise (habituation)
        let surprising = snapshot_with(0.8, 0.5, 0.5, 0.0);
        process_cycles(&mut dm, &surprising, 0, 20);

        assert!(
            dm.exploration_threshold > initial_threshold,
            "Threshold should adapt up on repeated surprise: {} > {}",
            dm.exploration_threshold,
            initial_threshold
        );
    }

    #[test]
    fn test_somatic_stress_dampens_exploration() {
        let mut dm = DriveManager::default();
        let mut stressed = snapshot_with(0.8, 0.5, 0.5, 0.0);
        stressed.somatic_stress = 0.8;

        let output_stressed = dm.process(&stressed);

        let mut dm2 = DriveManager::default();
        let unstressed = snapshot_with(0.8, 0.5, 0.5, 0.0);
        let output_normal = dm2.process(&unstressed);

        assert!(
            output_stressed.exploration_delta < output_normal.exploration_delta,
            "Stressed exploration should be dampened: {} < {}",
            output_stressed.exploration_delta,
            output_normal.exploration_delta
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let dm = DriveManager {
            boredom: 0.42,
            flow_intensity: 0.73,
            exploration_threshold: 0.55,
            in_flow: true,
            low_error_streak: 15,
            flow_streak: 8,
            ..Default::default()
        };

        let checkpoint = dm.checkpoint();
        let mut dm2 = DriveManager::default();
        dm2.restore(&checkpoint).unwrap();

        assert!((dm2.boredom - 0.42).abs() < 1e-6);
        assert!((dm2.flow_intensity - 0.73).abs() < 1e-6);
        assert!((dm2.exploration_threshold - 0.55).abs() < 1e-6);
        assert!(dm2.in_flow);
        assert_eq!(dm2.low_error_streak, 15);
        assert_eq!(dm2.flow_streak, 8);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut dm = DriveManager::default();
        let result = dm.restore(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_pe_does_not_change_drive_evidence_state() {
        let mut manager = DriveManager::default();
        let initial_threshold = manager.exploration_threshold;
        let mut snapshot = snapshot_with(1.0, 0.5, 0.5, 0.0);
        snapshot.prediction_error_valid = 0;

        let _ = manager.process(&snapshot);

        assert_eq!(manager.error_count, 0);
        assert_eq!(manager.low_error_streak, 0);
        assert_eq!(manager.flow_streak, 0);
        assert_eq!(manager.exploration_threshold, initial_threshold);
    }

    #[test]
    fn duplicate_and_out_of_order_pe_do_not_advance_drive_streaks() {
        let mut manager = DriveManager::default();
        let mut valid = snapshot_with(0.01, 0.8, 0.5, 0.0);
        valid.cycle_number = 10;
        manager.process(&valid);
        let initial_count = manager.error_count;
        let initial_low_streak = manager.low_error_streak;
        let initial_flow_streak = manager.flow_streak;
        let initial_threshold = manager.exploration_threshold;

        // A changed PE payload with the same cycle must be ignored.
        valid.prediction_error = 0.9;
        manager.process(&valid);
        // An older cycle must not rewind the evidence stream either.
        valid.cycle_number = 9;
        manager.process(&valid);

        assert_eq!(manager.error_count, initial_count);
        assert_eq!(manager.low_error_streak, initial_low_streak);
        assert_eq!(manager.flow_streak, initial_flow_streak);
        assert_eq!(manager.exploration_threshold, initial_threshold);
    }

    #[test]
    fn invalid_newer_pe_blocks_delayed_older_but_allows_same_cycle_recovery() {
        let mut manager = DriveManager::default();
        let mut invalid = snapshot_with(0.02, 0.8, 0.5, 0.0);
        invalid.cycle_number = 12;
        invalid.prediction_error_valid = 0;
        manager.process(&invalid);
        assert_eq!(manager.error_count, 0);
        assert_eq!(
            manager.prediction_error_cursor.last_observed_observation_cycle(),
            Some(12)
        );

        let mut delayed = snapshot_with(0.02, 0.8, 0.5, 0.0);
        delayed.cycle_number = 11;
        manager.process(&delayed);
        assert_eq!(manager.error_count, 0);

        let mut retry = delayed;
        retry.cycle_number = 12;
        manager.process(&retry);
        assert_eq!(manager.error_count, 1);
        manager.process(&retry);
        assert_eq!(manager.error_count, 1);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_drive_pe_watermarks() {
        let mut manager = DriveManager::default();
        let mut valid = snapshot_with(0.02, 0.8, 0.5, 0.0);
        valid.cycle_number = 10;
        manager.process(&valid);
        let mut invalid = valid;
        invalid.cycle_number = 12;
        invalid.prediction_error_valid = 0;
        manager.process(&invalid);

        let checkpoint = manager.checkpoint();
        assert_eq!(checkpoint.len(), 48);
        let mut restored = DriveManager::default();
        restored.restore(&checkpoint).unwrap();
        assert_eq!(
            restored.prediction_error_cursor.last_observed_observation_cycle(),
            Some(12)
        );
        assert_eq!(
            restored.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(10)
        );

        let mut delayed = valid;
        delayed.cycle_number = 11;
        restored.process(&delayed);
        assert_eq!(restored.error_count, 1);

        let mut retry = valid;
        retry.cycle_number = 12;
        restored.process(&retry);
        assert_eq!(restored.error_count, 2);
        restored.process(&retry);
        assert_eq!(restored.error_count, 2);
    }

    #[test]
    fn legacy_drive_checkpoint_restore_clears_cursor() {
        let mut manager = DriveManager::default();
        let mut prior = snapshot_with(0.3, 0.5, 0.5, 0.0);
        prior.cycle_number = 90;
        manager.process(&prior);
        let legacy = manager.checkpoint()[..25].to_vec();

        let mut restored = DriveManager::default();
        let mut later = prior;
        later.cycle_number = 100;
        restored.process(&later);
        restored.restore(&legacy).unwrap();
        assert_eq!(
            restored.prediction_error_cursor.last_observed_observation_cycle(),
            None
        );
        let mut cycle_one = prior;
        cycle_one.cycle_number = 1;
        restored.process(&cycle_one);
        assert_eq!(restored.error_count, 2); // ring buffer was not part of legacy state
    }

    #[test]
    fn malformed_drive_pe_checkpoint_is_rejected_atomically() {
        let mut manager = DriveManager::default();
        let mut sample = snapshot_with(0.2, 0.5, 0.5, 0.0);
        sample.cycle_number = 12;
        manager.process(&sample);
        let before_boredom = manager.boredom;
        let mut checkpoint = manager.checkpoint();
        checkpoint[0..4].copy_from_slice(&0.0_f32.to_le_bytes());
        checkpoint[40..48].copy_from_slice(&13_u64.to_le_bytes());
        assert!(manager.restore(&checkpoint).is_err());
        assert_eq!(manager.boredom, before_boredom);
        assert_eq!(
            manager.prediction_error_cursor.last_observed_observation_cycle(),
            Some(12)
        );
        assert_eq!(
            manager.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(12)
        );
    }

    #[test]
    fn test_neutral_output_on_neutral_input() {
        let mut dm = DriveManager::default();
        // Moderate error, moderate coherence = no strong drive signal
        let neutral = snapshot_with(0.2, 0.5, 0.5, 0.0);
        let output = dm.process(&neutral);

        // Should be close to neutral (small deltas only)
        assert!(
            output.exploration_delta.abs() < 0.1,
            "Neutral input should produce small exploration delta: {}",
            output.exploration_delta
        );
        assert!(
            (output.lr_modulation - 1.0).abs() < 0.2,
            "Neutral input should not drastically change LR: {}",
            output.lr_modulation
        );
    }
}
