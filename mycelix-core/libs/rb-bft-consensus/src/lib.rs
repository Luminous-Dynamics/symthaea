// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RB-BFT: Reputation-Based Byzantine Fault Tolerant Consensus
//!
//! This crate implements a Byzantine fault tolerant consensus protocol that uses
//! reputation-squared weighting to achieve 45% Byzantine tolerance (vs traditional 33%).
//!
//! ## Key Features
//!
//! - **Reputation² Weighting**: Votes are weighted by reputation squared, making it
//!   exponentially harder for low-reputation nodes to influence consensus.
//! - **45% Byzantine Tolerance**: The protocol can tolerate up to 45% malicious nodes
//!   when reputation is properly distributed.
//! - **K-Vector Integration**: Uses the 8-dimensional K-Vector trust representation.
//! - **Slashing Support**: Detects and penalizes Byzantine behavior.
//!
//! ## Protocol Overview
//!
//! 1. **Propose**: A leader proposes a block/value
//! 2. **Vote**: Validators vote with reputation-weighted signatures
//! 3. **Commit**: If weighted votes exceed threshold, consensus is reached
//! 4. **Challenge**: Any validator can challenge suspicious behavior

#[cfg(test)]
mod tests;

pub mod batch;
pub mod consensus;
pub mod crypto;
pub mod crypto_ops;
pub mod error;
pub mod network;
pub mod proposal;
pub mod round;
pub mod slashing;
pub mod validator;
pub mod vote;

// Advanced cryptographic modules
#[cfg(feature = "bls")]
pub mod bls;
#[cfg(feature = "pq")]
pub mod pq;
#[cfg(feature = "threshold")]
pub mod threshold;
pub mod vrf;

pub use batch::*;
pub use consensus::*;
pub use crypto::*;
pub use crypto_ops::*;
pub use error::*;
pub use network::*;
pub use proposal::*;
pub use round::*;
pub use slashing::*;
pub use validator::*;
pub use vote::*;

#[cfg(feature = "bls")]
pub use bls::*;
#[cfg(feature = "pq")]
pub use pq::*;
#[cfg(feature = "threshold")]
pub use threshold::*;
pub use vrf::*;

use mycelix_core_types::MAX_BYZANTINE_TOLERANCE;

/// Default timeout for consensus rounds (in milliseconds)
pub const DEFAULT_ROUND_TIMEOUT_MS: u64 = 30_000;

/// Minimum reputation to participate in consensus
pub const MIN_PARTICIPATION_REPUTATION: f32 = 0.1;

/// Minimum validators required for consensus
pub const MIN_VALIDATORS: usize = 5;

/// Re-export Byzantine tolerance from core types
pub const BYZANTINE_TOLERANCE: f32 = MAX_BYZANTINE_TOLERANCE;
