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

// ---------------------------------------------------------------------------
// Reciprocal-accountability threshold predicate (tag = 2)
// ---------------------------------------------------------------------------

/// Private/public inputs for the first SIF verifiable-computation circuit.
///
/// Only `private_value` remains secret. The guest commits the exact Mycelix
/// pre-attestation statement plus query/policy commitments and computes the
/// threshold predicate inside the zkVM. This is intentionally a narrow circuit:
/// it proves one real minimum-disclosure predicate rather than pretending a
/// generic digest wrapper proves arbitrary computation correctness.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AccountabilityThresholdInput {
    /// Sensitive scalar evaluated inside the zkVM and never journaled.
    pub private_value: u64,
    /// Public threshold used by the predicate `private_value >= threshold`.
    pub threshold: u64,
    /// Exact pre-attestation accountability receipt commitment from Mycelix.
    pub statement_digest: [u8; 32],
    /// Exact canonical query commitment.
    pub query_digest: [u8; 32],
    /// Exact policy commitment.
    pub policy_digest: [u8; 32],
    /// Public replay/domain nonce chosen for this logical operation.
    pub operation_nonce: [u8; 32],
}

/// Public journal for the SIF threshold predicate.
///
/// The private value is deliberately absent. A verifier learns only the bounded
/// boolean predicate and the commitments required to prove which accountability
/// statement/query/policy the computation belongs to.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AccountabilityThresholdOutput {
    /// Exact Mycelix pre-attestation receipt commitment proved by this execution.
    pub statement_digest: [u8; 32],
    /// Exact query commitment proved by this execution.
    pub query_digest: [u8; 32],
    /// Exact policy commitment proved by this execution.
    pub policy_digest: [u8; 32],
    /// Public threshold used by the verified predicate.
    pub threshold: u64,
    /// Verified result of `private_value >= threshold`.
    pub satisfied: bool,
    /// Public replay/domain nonce for the logical operation.
    pub operation_nonce: [u8; 32],
}
