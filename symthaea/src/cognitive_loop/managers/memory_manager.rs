// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Memory Manager — Episodic, Semantic, Resonator, Coordinator
//!
//! Consolidates memory-related logic into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Memory Systems Modeled
//!
//! 1. **Episodic consolidation**: High Phi moments get prioritized (Tulving 2002)
//! 2. **Semantic priming**: Contextual similarity boosts learning (Cowan 2001)
//! 3. **Consolidation pressure**: Memory load affects exploration/learning trade-off
//! 4. **Retrieval boost**: Strong recall boosts confidence
//!
//! ## Design
//!
//! The manager tracks memory utilization and consolidation pressure as internal state.
//! It does NOT own the actual memory stores (those remain in CognitiveLoopService).
//! Instead, it models the *meta-level* memory dynamics that affect the cognitive cycle.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thresholds;

/// Memory Manager — models meta-level memory dynamics.
///
/// Implements `CognitiveSubsystem` at interval 11 (co-prime).
pub struct MemoryManager {
    /// Rolling consolidation pressure [0, 1] — how full/active memory systems are
    consolidation_pressure: f32,
    /// Recent retrieval quality scores (ring buffer, capacity 8)
    retrieval_quality: [f32; 8],
    retrieval_idx: usize,
    retrieval_count: usize,
    /// Whether a consolidation event was recently triggered
    consolidation_active: bool,
    /// Cycles since last consolidation
    cycles_since_consolidation: u32,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self {
            consolidation_pressure: 0.0,
            retrieval_quality: [0.0; 8],
            retrieval_idx: 0,
            retrieval_count: 0,
            consolidation_active: false,
            cycles_since_consolidation: 0,
        }
    }
}

impl MemoryManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 11;

    /// Consolidation pressure threshold before requesting consolidation.
    /// Basis: Frankland & Bontempi (2005) — systems consolidation occurs under pressure.
    const CONSOLIDATION_THRESHOLD: f32 = 0.7;

    /// Pressure build rate per cycle from prediction error.
    const PRESSURE_BUILD_RATE: f32 = 0.01;

    /// Pressure decay rate per cycle (natural forgetting / integration).
    const PRESSURE_DECAY: f32 = 0.98;

    /// Cycles between forced consolidation events.
    /// Basis: Born & Wilhelm (2012) — sleep-dependent memory consolidation.
    const MAX_CONSOLIDATION_INTERVAL: u32 = 100;

    /// Current consolidation pressure [0.0, 1.0].
    /// High values indicate memory systems are overloaded and need dreaming/consolidation.
    pub fn consolidation_pressure(&self) -> f32 {
        self.consolidation_pressure
    }

    /// Mean retrieval quality [0.0, 1.0].
    /// High values indicate reliable memory recall; low values suggest retrieval failures.
    pub fn recall_quality(&self) -> f32 {
        self.mean_retrieval_quality()
    }

    fn mean_retrieval_quality(&self) -> f32 {
        if self.retrieval_count == 0 {
            return 0.5;
        }
        let sum: f32 = self.retrieval_quality[..self.retrieval_count].iter().sum();
        sum / self.retrieval_count as f32
    }

    fn push_retrieval(&mut self, quality: f32) {
        self.retrieval_quality[self.retrieval_idx] = quality;
        self.retrieval_idx = (self.retrieval_idx + 1) % 8;
        if self.retrieval_count < 8 {
            self.retrieval_count += 1;
        }
    }
}

impl CognitiveSubsystem for MemoryManager {
    fn name(&self) -> &'static str {
        "memory_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;
        self.cycles_since_consolidation += 1;

        // ── 1. Update consolidation pressure ──────────────────────────────
        // Prediction error generates encoding demand → pressure builds
        let pe = snapshot.prediction_error;
        self.consolidation_pressure += pe * Self::PRESSURE_BUILD_RATE;
        // Natural decay (forgetting curve integration)
        self.consolidation_pressure *= Self::PRESSURE_DECAY;
        self.consolidation_pressure = self.consolidation_pressure.clamp(0.0, 1.0);

        // ── 2. Consolidation triggering ───────────────────────────────────
        if self.consolidation_pressure > Self::CONSOLIDATION_THRESHOLD
            || self.cycles_since_consolidation > Self::MAX_CONSOLIDATION_INTERVAL
        {
            self.consolidation_active = true;
            self.cycles_since_consolidation = 0;
            // During consolidation: dampen exploration, boost learning
            output.exploration_delta -= thresholds::MEMORY_CONSOLIDATION_EXPLORATION_DAMPEN;
            output.lr_modulation = thresholds::MEMORY_CONSOLIDATION_LR_BOOST;
            output.flags |= output_flags::REQUEST_CONSOLIDATION;
        } else {
            self.consolidation_active = false;
        }

        // ── 3. Retrieval quality → confidence ─────────────────────────────
        // Good retrieval (high coherence + prediction confidence) signals reliable memory
        let retrieval_signal = (snapshot.prediction_confidence
            * thresholds::MEMORY_RETRIEVAL_CONFIDENCE_WEIGHT
            + f64::from(snapshot.coherence) * thresholds::MEMORY_RETRIEVAL_COHERENCE_WEIGHT)
            .clamp(0.0, 1.0) as f32;
        self.push_retrieval(retrieval_signal);

        let mean_quality = self.mean_retrieval_quality();
        if mean_quality > thresholds::MEMORY_RETRIEVAL_HIGH_QUALITY {
            // Strong retrieval → confidence boost
            output.confidence_delta += (mean_quality - thresholds::MEMORY_RETRIEVAL_HIGH_QUALITY)
                as f64
                * thresholds::MEMORY_RETRIEVAL_CONFIDENCE_GAIN;
        } else if mean_quality < thresholds::MEMORY_RETRIEVAL_LOW_QUALITY {
            // Poor retrieval → signal exploration need
            output.exploration_delta += (thresholds::MEMORY_RETRIEVAL_LOW_QUALITY - mean_quality)
                as f64
                * thresholds::MEMORY_RETRIEVAL_EXPLORATION_GAIN;
        }

        // ── 4. High consciousness → episodic consolidation boost ──────────
        // Phi-weighted moments deserve priority encoding
        if snapshot.unified_psi > thresholds::MEMORY_PSI_CONSOLIDATION_THRESHOLD {
            output.flags |= output_flags::REQUEST_CONSOLIDATION;
            output.lr_modulation *= 1.0
                + (snapshot.unified_psi - thresholds::MEMORY_PSI_CONSOLIDATION_THRESHOLD)
                    * thresholds::MEMORY_PSI_LR_SCALE;
        }

        // ── 5. Memory load → exploration gate ─────────────────────────────
        // High consolidation pressure means "digest before exploring"
        if self.consolidation_pressure > thresholds::MEMORY_PRESSURE_EXPLORATION_THRESHOLD as f32 {
            let dampen = (self.consolidation_pressure as f64
                - thresholds::MEMORY_PRESSURE_EXPLORATION_THRESHOLD)
                * thresholds::MEMORY_PRESSURE_EXPLORATION_DAMPEN;
            output.exploration_delta -= dampen;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&self.consolidation_pressure.to_le_bytes());
        data.extend_from_slice(&self.cycles_since_consolidation.to_le_bytes());
        data.push(self.consolidation_active as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 9 {
            return Err(format!(
                "MemoryManager checkpoint too short: {} < 9",
                data.len()
            ));
        }
        self.consolidation_pressure = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "MemoryManager: corrupt checkpoint bytes [0..4]".to_string())?,
        );
        self.cycles_since_consolidation = u32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "MemoryManager: corrupt checkpoint bytes [4..8]".to_string())?,
        );
        self.consolidation_active = data[8] != 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn snapshot(pe: f32, conf: f32, coherence: f32, psi: f32) -> CycleSnapshot {
        CycleSnapshot {
            prediction_error: pe,
            prediction_confidence: f64::from(conf),
            coherence,
            unified_psi: f64::from(psi),
            ..Default::default()
        }
    }

    #[test]
    fn test_defaults() {
        let mm = MemoryManager::default();
        assert_eq!(mm.consolidation_pressure, 0.0);
        assert!(!mm.consolidation_active);
    }

    #[test]
    fn test_name_and_interval() {
        let mm = MemoryManager::default();
        assert_eq!(mm.name(), "memory_manager");
        assert_eq!(mm.interval(), 11);
    }

    #[test]
    fn test_pressure_builds_on_error() {
        let mut mm = MemoryManager::default();
        let high_error = snapshot(0.8, 0.5, 0.5, 0.4);

        for _ in 0..50 {
            mm.process(&high_error);
        }

        assert!(
            mm.consolidation_pressure > 0.0,
            "Pressure should build: {}",
            mm.consolidation_pressure
        );
    }

    #[test]
    fn test_consolidation_triggers() {
        let mut mm = MemoryManager {
            consolidation_pressure: 0.8,
            ..Default::default()
        };

        let s = snapshot(0.5, 0.5, 0.5, 0.4);
        let output = mm.process(&s);

        assert!(mm.consolidation_active);
        assert!(output.flags & output_flags::REQUEST_CONSOLIDATION != 0);
    }

    #[test]
    fn test_timed_consolidation() {
        let mut mm = MemoryManager {
            cycles_since_consolidation: 101,
            ..Default::default()
        };

        let s = snapshot(0.01, 0.5, 0.5, 0.4);
        let output = mm.process(&s);

        assert!(mm.consolidation_active);
        assert!(output.flags & output_flags::REQUEST_CONSOLIDATION != 0);
    }

    #[test]
    fn test_high_psi_boosts_consolidation() {
        let mut mm = MemoryManager::default();
        let high_psi = snapshot(0.1, 0.5, 0.5, 0.8);

        let output = mm.process(&high_psi);
        assert!(
            output.flags & output_flags::REQUEST_CONSOLIDATION != 0,
            "High Psi should request consolidation"
        );
        assert!(
            output.lr_modulation > 1.0,
            "High Psi should boost learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_good_retrieval_boosts_confidence() {
        let mut mm = MemoryManager::default();
        let good = snapshot(0.1, 0.9, 0.9, 0.5);

        // Fill retrieval buffer
        for _ in 0..10 {
            mm.process(&good);
        }

        let output = mm.process(&good);
        assert!(
            output.confidence_delta > 0.0,
            "Good retrieval should boost confidence: {}",
            output.confidence_delta
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mm = MemoryManager {
            consolidation_pressure: 0.65,
            cycles_since_consolidation: 42,
            consolidation_active: true,
            ..Default::default()
        };

        let data = mm.checkpoint();
        let mut mm2 = MemoryManager::default();
        mm2.restore(&data).unwrap();

        assert!((mm2.consolidation_pressure - 0.65).abs() < 1e-6);
        assert_eq!(mm2.cycles_since_consolidation, 42);
        assert!(mm2.consolidation_active);
    }
}
