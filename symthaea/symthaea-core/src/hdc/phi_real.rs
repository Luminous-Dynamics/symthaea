//! # DEPRECATED - Use `spectral_connectivity` module instead
//!
//! This module has been renamed to `spectral_connectivity` to accurately
//! reflect what it computes: **algebraic connectivity (λ₂)**, NOT IIT Φ.
//!
//! ## Migration Guide
//!
//! ```rust,ignore
//! // OLD (deprecated):
//! use symthaea::hdc::phi_real::RealPhiCalculator;
//! let phi = calculator.compute(&components);
//!
//! // NEW (correct):
//! use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;
//! let lambda2 = calculator.algebraic_connectivity(&components);
//! ```
//!
//! ## Why This Change?
//!
//! Validation revealed λ₂ has **r = -0.62 correlation** with IIT Φ - they measure
//! nearly opposite properties! See `docs/METRIC_CLARIFICATION.md` for details.
//!
//! ## Valid Use Cases for λ₂
//!
//! ✅ Network connectivity analysis
//! ✅ Synchronization potential estimation
//! ✅ Graph mixing properties
//!
//! ## Invalid Use Cases
//!
//! ❌ Consciousness measurement claims
//! ❌ IIT Φ approximation
//! ❌ Integrated information estimation

#![deprecated(since = "0.5.0", note = "Module renamed to spectral_connectivity - this measures λ₂, NOT IIT Φ. See docs/METRIC_CLARIFICATION.md")]

// Re-export everything from spectral_connectivity for backward compatibility
pub use crate::hdc::spectral_connectivity::*;
