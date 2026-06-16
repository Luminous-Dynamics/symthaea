// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Response types matching the Mycelix SDK for conductor zome call results.

use serde::{Deserialize, Serialize};

/// SAP currency balance for a member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceResponse {
    pub member_did: String,
    pub currency: String,
    pub balance: u64,
    pub available: bool,
}

/// TEND time-currency balance for a member, including MYCEL score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TendBalanceResponse {
    pub member_did: String,
    pub balance: i32,
    pub mycel_score: f64,
    pub available: bool,
}

/// Fee tier information derived from MYCEL score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTierResponse {
    pub member_did: String,
    pub mycel_score: f64,
    pub tier_name: String,
    pub base_fee_rate: f64,
}

/// Collateral deposit with optional ZK balance proof.
///
/// When `balance_proof` is `Some`, it contains the serialized RISC Zero
/// receipt bytes proving that the depositor holds at least
/// `collateral_amount` without revealing their actual balance.
/// The bridge can forward these opaque bytes to any verifier that has
/// the guest image ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralDepositWithProof {
    pub depositor_did: String,
    pub collateral_type: String,
    pub collateral_amount: u64,
    pub oracle_rate: f64,
    /// Optional ZK proof that depositor has sufficient collateral.
    /// Serialized receipt bytes (opaque to the bridge).
    pub balance_proof: Option<Vec<u8>>,
}
