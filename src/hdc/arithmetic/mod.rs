// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hyperdimensional Arithmetic Engine
//!
//! **Modular architecture for mathematical cognition**
//!
//! This module provides a decomposed view of the arithmetic engine:
//!
//! ## Modules
//!
//! - `numbers` - Peano number construction (`HdcNumber`)
//! - `operations` - Core arithmetic engine (`ArithmeticEngine`, `ArithmeticResult`)
//! - `theorem_prover` - Mathematical theorem proving
//! - `hybrid` - Hybrid architecture for efficiency (`HybridArithmeticEngine`)
//! - `reasoning` - Math-logic reasoning bridge (`MathReasoningBridge`)
//! - `symbolic` - Symbolic algebra (`SymbolicExpr`, `Polynomial`)
//! - `multi_path` - Multi-path proof verification (`MultiPathVerifier`)
//!
//! ## Backward Compatibility
//!
//! All types are re-exported from the parent module for compatibility:
//! ```rust,ignore
//! use symthaea::hdc::arithmetic_engine::*; // Still works
//! use symthaea::hdc::arithmetic::*;        // New recommended path
//! ```

// Re-export everything from the monolithic file for backward compatibility
// This allows gradual migration to the modular structure
pub use super::arithmetic_engine::*;

// Future module structure (to be implemented incrementally):
//
// mod numbers;
// mod operations;
// mod theorem_prover;
// mod hybrid;
// mod reasoning;
// mod symbolic;
// mod multi_path;
//
// pub use numbers::*;
// pub use operations::*;
// pub use theorem_prover::*;
// pub use hybrid::*;
// pub use reasoning::*;
// pub use symbolic::*;
// pub use multi_path::*;
