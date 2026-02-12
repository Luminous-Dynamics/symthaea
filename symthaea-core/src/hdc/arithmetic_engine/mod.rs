//! # Hyperdimensional Arithmetic Engine
//!
//! Mathematical cognition from first principles using Hyperdimensional Computing.
//!
//! ## Module Organization
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`verification`] | Adaptive HDC similarity thresholds for result verification |
//! | [`core_number`] | `HdcNumber` — Peano-constructed numbers encoded as `BinaryHV` |
//! | [`engine`] | `ArithmeticEngine` — add, multiply, subtract, power, factorial via Peano axioms |
//! | [`theorems`] | `TheoremProver` — proves commutativity, associativity, distributivity |
//! | [`hybrid`] | `HybridArithmeticEngine` — deep (Peano) + fast (direct) path dispatcher |
//! | [`discovery`] | `MathDiscovery` — Phi-guided proof exploration and conjecture generation |
//! | [`reasoning_bridge`] | `MathReasoningBridge` — bridges arithmetic proofs to logical inference |
//! | [`symbolic`] | Symbolic algebra: expressions, polynomials, simplification, equation solving |
//! | [`multi_path`] | `MultiPathVerifier` — multiple independent proof strategies with Phi ranking |

mod verification;
mod core_number;
mod engine;
mod theorems;
mod hybrid;
mod discovery;
mod reasoning_bridge;
mod symbolic;
mod multi_path;

#[cfg(test)]
mod tests;

// Re-export everything for backward compatibility
pub use verification::*;
pub use core_number::*;
pub use engine::*;
pub use theorems::*;
pub use hybrid::*;
pub use discovery::*;
pub use reasoning_bridge::*;
pub use symbolic::*;
pub use multi_path::*;
