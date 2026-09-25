// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Productive-capability closure contracts plus deterministic structural analysis.
//!
//! The CIV-BOOT-002A contract remains byte-for-byte in `lib.rs`; this root
//! composes it with the CIV-BOOT-002B/002C evaluators without giving the analysis
//! crate any fabrication, network, filesystem, HAL, or resource-allocation authority.

#![deny(unsafe_code)]

#[path = "lib.rs"]
mod contract;
pub use contract::*;

mod reachability;
pub use reachability::{evaluate_reachability, ReachabilityReportV1};

mod import_leverage;
pub use import_leverage::evaluate_import_leverage;
