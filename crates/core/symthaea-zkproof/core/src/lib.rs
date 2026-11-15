#![cfg_attr(not(feature = "std"), no_std)]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

extern crate alloc;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// Data passed from host to guest (zkVM)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionInput {
    pub episodes: Vec<Vec<f32>>, // HDC vectors (e.g., 1024D)
    pub tau_scale: f32,
    pub threshold: f32,
}

/// Data committed by the guest as public output
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionOutput {
    pub average_phi: f32,
    pub tau_scale: f32,
    pub episode_count: u32,
}

// ---------------------------------------------------------------------------
// Balance proof types (tag = 1)
// ---------------------------------------------------------------------------

/// Private input for a ZK balance sufficiency proof.
///
/// The `balance` field is PRIVATE — it is consumed inside the zkVM guest
/// but never appears in the committed journal output.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BalanceProofInput {
    /// Actual balance (PRIVATE — never revealed in the proof output).
    pub balance: u64,
    /// Minimum balance required (will be committed as public output).
    pub required_minimum: u64,
    /// Caller-chosen nonce to prevent proof replay (public output).
    pub nonce: u64,
}

/// Public output committed by the guest for a balance proof.
///
/// The actual balance is NOT included — only the boolean result,
/// the threshold, and the nonce are revealed.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BalanceProofOutput {
    /// Whether `balance >= required_minimum`.
    pub sufficient: bool,
    /// The minimum that was checked against (public).
    pub required_minimum: u64,
    /// Replay-prevention nonce (public).
    pub nonce: u64,
}
