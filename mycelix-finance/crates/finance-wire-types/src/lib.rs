// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wire types for Mycelix Finance.
//!
//! External-facing types for cross-cluster calls and test fixtures.
//! These mirror the integrity entry types but are usable without HDK dependencies.

use serde::{Deserialize, Serialize};

/// Asset type for collateral registration.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssetType {
    RealEstate,
    Vehicle,
    Cryptocurrency,
    EnergyAsset,
    Equipment,
    Other(String),
}

/// Input for registering collateral against a loan or stake.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterCollateralInput {
    pub owner_did: String,
    pub asset_type: AssetType,
    pub description: String,
    pub estimated_value_sap: u64,
    pub property_id: Option<String>,
}

/// Response for SAP balance queries.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SapBalanceResponse {
    pub did: String,
    pub balance: u64,
    pub pending_in: u64,
    pub pending_out: u64,
}
