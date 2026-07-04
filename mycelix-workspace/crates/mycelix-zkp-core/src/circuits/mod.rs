// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared ZKP circuits for the Mycelix ecosystem.
//!
//! These circuits are used by multiple clusters:
//! - range_proof: Health (VitalsInRange, AgeRange), Finance (balance_range)

#[cfg(feature = "backend-winterfell")]
pub mod range_proof;

#[cfg(feature = "backend-winterfell")]
pub mod jurisdiction_proof;

#[cfg(feature = "backend-winterfell")]
pub mod review_integrity;

#[cfg(feature = "backend-winterfell")]
pub mod recursive_aggregation;

pub mod merkle_membership;
pub mod nullifier;

#[cfg(feature = "backend-winterfell")]
pub mod winterfell_bench;

#[cfg(feature = "backend-winterfell")]
pub mod winterfell_xor;
