//! # Unified Consciousness Engine
//!
//!
//! Wraps the 4 independent consciousness measurement systems into a single
//! coherent engine with clear data flow and unified output.
//!
//! ## Systems (Tiered Architecture)
//!
//! | Tier | System | Interval | Science |
//! |------|--------|----------|---------|
//! | Layer 1 | SpectralMIPFinder | 97 cycles | Tononi (2004) — IIT Phi via Fiedler ordering |
//! | Layer 2 | MultiModalIntegrator | 13 cycles | Damasio (1994) — cross-modal binding Phi |
//! | Layer 3 | ConsciousnessEquationV2 | 23 cycles | 7-theory master equation C(t) |
//! | Layer 4 | UnifiedConsciousnessPipeline | 97 cycles | Dehaene (2011) — end-to-end pipeline |
//!
//! ## Design Principles
//!
//! 1. **Single entry point**: `measure()` called once per cycle
//! 2. **No direct field mutation**: Returns `ConsciousnessEngineOutput` with proposed deltas
//! 3. **Preserves co-prime intervals**: Each subsystem fires at its original rate
//! 4. **Backward compatible**: All existing carryover fields populated
//! 5. **WASM-ready**: Output struct uses only Copy/Clone types (no trait objects)

mod types;
mod measure;
mod helpers;

pub(crate) use types::{
    ConsciousnessEngineCache, ConsciousnessEngineInput, ConsciousnessEngineOutput,
    ConsciousnessWeights, MoralConsciousnessCoupling, WeightConvergenceState,
};

use symthaea_core::consciousness_metrics::SpectralMIPFinder;
use crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2;
use crate::consciousness::multi_modal_integration::MultiModalIntegrator;
use crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline;

/// The unified consciousness measurement engine.
///
/// Owns all 4 measurement systems and orchestrates their execution
/// at co-prime intervals.
pub(crate) struct ConsciousnessEngine {
    // ── Layer 1: Spectral MIP (always present) ─────────────────────────
    spectral_mip_finder: SpectralMIPFinder,

    // ── Layer 2: Multi-modal integration (optional) ────────────────────
    multi_modal_integrator: Option<MultiModalIntegrator>,

    // ── Layer 3: Master equation (optional) ────────────────────────────
    consciousness_equation_v2: Option<ConsciousnessEquationV2>,

    // ── Layer 4: Full pipeline (optional) ──────────────────────────────
    unified_consciousness_pipeline: Option<UnifiedConsciousnessPipeline>,

    // ── Moral coupling ─────────────────────────────────────────────────
    moral_coupling: MoralConsciousnessCoupling,

    // ── Cached values (for non-firing cycles) ──────────────────────────
    cache: ConsciousnessEngineCache,
}

impl ConsciousnessEngine {
    /// Create a new consciousness engine from its component systems.
    pub fn new(
        spectral_mip_finder: SpectralMIPFinder,
        multi_modal_integrator: Option<MultiModalIntegrator>,
        consciousness_equation_v2: Option<ConsciousnessEquationV2>,
        unified_consciousness_pipeline: Option<UnifiedConsciousnessPipeline>,
    ) -> Self {
        Self {
            spectral_mip_finder,
            multi_modal_integrator,
            consciousness_equation_v2,
            unified_consciousness_pipeline,
            moral_coupling: MoralConsciousnessCoupling::default(),
            cache: ConsciousnessEngineCache::default(),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
