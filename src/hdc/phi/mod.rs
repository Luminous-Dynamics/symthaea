//! Phi (Φ) - Integrated Information Theory calculations
//!
//! Re-exports phi-related types from symthaea-core.

// Re-export phi types from symthaea-core
pub use symthaea_core::phi_engine::{
    PhiEngine,
    PhiMethod,
    PhiResult,
    PhiCalculator,
    ContinuousPhiCalculator,
    TieredPhi,
    ApproximationTier,
    TieredPhiConfig,
    CachedPhiEngine,
    CacheStats,
};
