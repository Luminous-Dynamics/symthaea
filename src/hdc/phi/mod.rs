//! Phi (Φ) - Integrated Information Theory calculations
//!
//! Re-exports phi-related types from symthaea-core.

// Re-export phi types from symthaea-core
pub use symthaea_core::phi_engine::{
    ApproximationTier, CacheStats, CachedPhiEngine, ContinuousPhiCalculator, PhiCalculator,
    PhiEngine, PhiMethod, PhiResult, TieredPhi, TieredPhiConfig,
};
