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
        }
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
    }
}
