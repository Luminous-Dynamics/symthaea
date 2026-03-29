// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Winterfell AIR for PoGQ v4.1 Byzantine Detection
//!
//! This module implements a specialized AIR (Algebraic Intermediate Representation)
//! for proving correct execution of the PoGQ state machine. It targets 3-10× faster
//! proving than the RISC Zero zkVM approach by using a domain-specific AIR instead
//! of general-purpose VM execution.
//!
//! ## Performance Target
//! - **Proving time**: 5-15 seconds (vs 46.6s RISC Zero)
//! - **Verification**: <50ms (vs 92ms RISC Zero)
//! - **Proof size**: ~200KB (similar to RISC Zero)
//!
//! ## AIR Design
//!
//! The PoGQ state machine has 8 registers tracked across execution:
//!
//! ### Execution Trace Layout
//! ```text
//! | Cycle | ema_t | viol_t | clear_t | quar_t | x_t | threshold | beta | round |
//! |-------|-------|--------|---------|--------|-----|-----------|------|-------|
//! | 0     | init  | 0      | 0       | 0      | ... | ...       | ...  | 0     |
//! | 1     | ...   | ...    | ...     | ...    | ... | ...       | ...  | 1     |
//! ```
//!
//! ### Transition Constraints
//!
//! For each row t → t+1:
//!
//! 1. **EMA Update** (Q16.16 fixed-point):
//!    ```text
//!    ema[t+1] = (beta * ema[t] + (65536 - beta) * x[t]) / 65536
//!    ```
//!
//! 2. **Violation Flag**:
//!    ```text
//!    is_viol = (x[t] < threshold) ? 1 : 0
//!    ```
//!
//! 3. **Counter Updates**:
//!    ```text
//!    viol[t+1] = is_viol ? viol[t] + 1 : 0
//!    clear[t+1] = !is_viol ? clear[t] + 1 : 0
//!    ```
//!
//! 4. **Hysteresis Logic**:
//!    ```text
//!    entering = (viol[t+1] >= k)
//!    releasing = (quar[t] == 1) && (clear[t+1] >= m)
//!    quar[t+1] = (round[t+1] <= w) ? 0 : (releasing ? 0 : (entering ? 1 : quar[t]))
//!    ```
//!
//! ### Boundary Constraints
//!
//! - Initial state (row 0): `ema[0] = init_ema`, `viol[0] = 0`, `clear[0] = 0`, `quar[0] = 0`
//! - Final state (row N): `quar[N]` is the public output
//!
//! ### Public Inputs
//!
//! - PoGQ parameters: `beta`, `w`, `k`, `m`, `threshold`
//! - Initial state: `ema[0]`, `viol[0]`, `clear[0]`, `quar[0]`, `round[0]`
//! - Expected output: `quar[N]`
//!
//! ### Private Witness
//!
//! - Hybrid scores: `x[0..N]` (one per round)

mod air;
mod prover;
mod provenance;
mod security;
mod trace;

pub use air::{PoGQAir, PublicInputs as AirPublicInputs, AIR_SCHEMA_REV, SCALE};
pub use prover::PoGQProver;
pub use provenance::ProvenanceHash;
pub use security::{SecurityProfile, PerformanceEstimate};
pub use trace::TraceBuilder;

// Python bindings (conditional compilation)
#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests;
