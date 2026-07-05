// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cantor Dream Manager — fractal broadcast buffer + cleanup engine.
//!
//! Groups GWT→Cantor broadcast buffer, persistent cleanup engine,
//! activation tracking, dream surprise EMA, and resonance boost into
//! a single data-holder manager.
//!
//! Science: Baars (1988) + Stickgold (2005) — conscious broadcast → fractal dreaming;
//!          Born & Wilhelm (2012) — sleep spindle replay strengthens stable traces.

use symthaea_core::hdc::cantor_recursive_hv::CantorRecursiveHV;
use symthaea_core::hdc::cantor_resonator_cleanup::CantorCleanupEngine;

/// Sleep phase for NREM/REM alternation.
///
/// Science: Born & Wilhelm (2012) — NREM consolidates specific memories via
/// hippocampal-cortical replay; REM integrates across episodes and processes
/// emotional content. WSCL framework (2024): alternating prevents catastrophic
/// forgetting by interleaving recent and remote memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DreamPhase {
    /// NREM-like: Replay recent episodes interleaved with old ones.
    /// Strengthens specific memories. High consolidation LR.
    /// Prioritizes episodes with high ripple amplitude (reward + surprise).
    Nrem,
    /// REM-like: Cross-episode pattern extraction and emotional reprocessing.
    /// Finds abstract structure shared across episodes. Lower LR, broader sampling.
    /// Prioritizes high-valence episodes and cross-episode similarity.
    Rem,
}

/// NREM/REM alternation constants.
/// Biological 90-min cycle scaled to dream interval (~20 cycles).
const NREM_DURATION: u32 = 15; // 75% NREM (3:1 ratio matches biology)
const REM_DURATION: u32 = 5; // 25% REM

/// Consolidated Cantor dream subsystem.
///
/// Holds fractal broadcast, cleanup engine, and dream telemetry fields
/// that were previously scattered across `CognitiveLoopService`.
pub(crate) struct CantorDreamManager {
    /// CRHVs created from GWT broadcasts for dream consolidation.
    /// When a thought becomes "conscious" (enters workspace and is broadcast), it gets
    /// wrapped as a Cantor Recursive Hypervector preserving multi-scale structure.
    /// During dream consolidation, the CantorCleanupEngine factorizes these through
    /// the resonator codebook, preventing metacognitive amnesia.
    /// Science: Baars (1988) + Stickgold (2005) — conscious broadcast → fractal dreaming.
    pub(crate) broadcast_buffer: Vec<CantorRecursiveHV>,

    /// Persistent Cantor cleanup engine: codebook accumulates across dream cycles.
    /// Unlike the previous ephemeral approach (rebuild each dream), this engine retains
    /// learned representations so dream consolidation genuinely strengthens memories
    /// over the brain's lifetime.
    /// Science: Born & Wilhelm (2012) — sleep spindle replay strengthens stable traces;
    ///          Walker (2009) — offline consolidation requires persistent memory stores.
    pub(crate) cleanup_engine: CantorCleanupEngine,

    /// Last GWT activation strength, used for adaptive CRHV depth.
    /// Stronger activations (higher workspace competition score) produce deeper fractals.
    /// Science: Dehaene et al. (2006) — ignition strength varies with stimulus salience.
    pub(crate) last_activation: f32,

    /// EMA of dream consolidation surprise (|pre_ss − post_ss|).
    /// High surprise signals the codebook is encountering novel fractal structure.
    /// Science: Friston (2010) — free-energy surprise drives plasticity updates.
    pub(crate) dream_surprise: f32,

    /// Resonance boost from coherent CRHV pairs in the broadcast buffer.
    /// When multiple CRHVs share high similarity (>0.8), the resulting coalition
    /// amplifies workspace integration — a "fractal choir" effect.
    /// Science: Edelman & Tononi (2000) — reentrant cortical signaling;
    ///          Singer (1999) — binding by synchrony.
    pub(crate) resonance_boost: f32,

    /// Current sleep phase (NREM or REM).
    pub(crate) dream_phase: DreamPhase,

    /// Counter within current phase (resets on phase transition).
    pub(crate) phase_counter: u32,

    /// Total dream cycles completed (for phase alternation).
    pub(crate) total_dream_cycles: u64,
}

impl CantorDreamManager {
    /// Create a new CantorDreamManager with default state.
    ///
    /// `dim` is the HDC dimension used for the cleanup engine codebook.
    pub fn new(dim: usize) -> Self {
        use symthaea_core::hdc::cantor_resonator_cleanup::*;
        Self {
            broadcast_buffer: Vec::with_capacity(32),
            cleanup_engine: CantorCleanupEngine::with_codebook_capacity(dim),
            last_activation: 0.0,
            dream_surprise: 0.0,
            resonance_boost: 0.0,
            dream_phase: DreamPhase::Nrem,
            phase_counter: 0,
            total_dream_cycles: 0,
        }
    }

    /// Advance dream phase and return the current phase.
    ///
    /// Called once per dream cycle. Alternates NREM→REM at 3:1 ratio
    /// (15 NREM cycles, then 5 REM cycles, repeat).
    ///
    /// Science: Diekelmann & Born (2010) — NREM strengthens specific traces,
    /// REM integrates across episodes preventing catastrophic forgetting.
    pub fn advance_phase(&mut self) -> DreamPhase {
        self.phase_counter += 1;
        self.total_dream_cycles += 1;

        let duration = match self.dream_phase {
            DreamPhase::Nrem => NREM_DURATION,
            DreamPhase::Rem => REM_DURATION,
        };

        if self.phase_counter >= duration {
            self.phase_counter = 0;
            self.dream_phase = match self.dream_phase {
                DreamPhase::Nrem => DreamPhase::Rem,
                DreamPhase::Rem => DreamPhase::Nrem,
            };
        }

        self.dream_phase
    }

    /// Whether current phase is REM (cross-episode integration).
    pub fn is_rem(&self) -> bool {
        self.dream_phase == DreamPhase::Rem
    }

    /// Compute adaptive dream consolidation interval based on current learning rate boost.
    ///
    /// When `lr_boost` is high (system is learning rapidly), the interval shortens so
    /// consolidation keeps pace with incoming experience. At low LR, consolidation
    /// is infrequent — the system is in a stable state with little to integrate.
    ///
    /// Formula: `max(DREAM_MIN_INTERVAL, DREAM_BASE_INTERVAL / (1 + DREAM_LR_INTERVAL_SCALE * |lr_boost|))`
    ///
    /// Science: Diekelmann & Born (2010) — sleep consolidation scales with learning load;
    ///          Walker (2017) — consolidation need correlates with encoding intensity.
    pub fn adaptive_interval(&self, lr_boost: f64) -> u64 {
        use super::thresholds::{DREAM_BASE_INTERVAL, DREAM_LR_INTERVAL_SCALE, DREAM_MIN_INTERVAL};
        let divisor = 1.0 + DREAM_LR_INTERVAL_SCALE * lr_boost.abs();
        let interval = (DREAM_BASE_INTERVAL as f64 / divisor).round() as u64;
        interval.max(DREAM_MIN_INTERVAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cantor_dream_manager_new() {
        let mgr = CantorDreamManager::new(128);
        assert!(mgr.broadcast_buffer.is_empty());
        assert_eq!(mgr.last_activation, 0.0);
        assert_eq!(mgr.dream_surprise, 0.0);
        assert_eq!(mgr.resonance_boost, 0.0);
        assert_eq!(mgr.dream_phase, DreamPhase::Nrem);
        assert_eq!(mgr.phase_counter, 0);
    }

    #[test]
    fn test_nrem_rem_alternation() {
        let mut mgr = CantorDreamManager::new(128);

        // Collect 25 phases to see the full NREM→REM→NREM cycle
        let phases: Vec<DreamPhase> = (0..25).map(|_| mgr.advance_phase()).collect();

        // Count NREM and REM in first 20 cycles (one full period)
        let nrem_count = phases[..20]
            .iter()
            .filter(|&&p| p == DreamPhase::Nrem)
            .count();
        let rem_count = phases[..20]
            .iter()
            .filter(|&&p| p == DreamPhase::Rem)
            .count();

        // 3:1 ratio: ~15 NREM, ~5 REM (±1 for boundary)
        assert!(
            nrem_count >= 14 && nrem_count <= 16,
            "NREM count {nrem_count} should be ~15"
        );
        assert!(
            rem_count >= 4 && rem_count <= 6,
            "REM count {rem_count} should be ~5"
        );

        // Must contain both phases
        assert!(phases.contains(&DreamPhase::Nrem));
        assert!(phases.contains(&DreamPhase::Rem));

        // Total cycles tracked
        assert_eq!(mgr.total_dream_cycles, 25);
    }

    #[test]
    fn test_rem_detection() {
        let mut mgr = CantorDreamManager::new(128);
        assert!(!mgr.is_rem());

        // Advance through NREM
        for _ in 0..15 {
            mgr.advance_phase();
        }
        assert!(mgr.is_rem());
    }
}
