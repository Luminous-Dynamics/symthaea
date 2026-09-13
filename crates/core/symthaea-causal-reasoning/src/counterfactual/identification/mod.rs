// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Identification
//!
//! Implements Pearl-style causal identification primitives:
//! - Rule 1: insertion/deletion of observations (backdoor/frontdoor criteria)
//! - Rule 2: action/observation exchange
//! - Rule 3: insertion/deletion of actions
//! - exact truncated-factorization identification for qualified fully observed Markovian DAGs
//!
//! ## Qualification boundary
//!
//! [`QualifiedMarkovianId`] is the current claim-safe exact-estimand path. It validates the DAG
//! and query, implements the Markovian g-formula, and fails closed on bidirected/latent graphs.
//!
//! [`IDAlgorithm`] remains the broader experimental latent-variable implementation. Its complete
//! Shpitser-Pearl ID claim is **not established** by the current RQ-005 evidence program; callers
//! requiring qualified causal authority should not treat a successful legacy `IDAlgorithm`
//! result as evidence of complete latent-variable identification.

pub mod dag;
pub mod discovery;
pub mod estimation;
pub mod id_algorithm;
pub mod qualified_markovian;
pub mod reasoner;

#[cfg(test)]
mod tests;

// Re-export everything for backward compatibility.
pub use dag::*;
pub use discovery::*;
pub use estimation::*;
pub use id_algorithm::*;
pub use qualified_markovian::*;
pub use reasoner::*;
