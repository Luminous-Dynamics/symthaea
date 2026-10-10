// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Systemic-analysis authority primitives.
//!
//! This first tranche deliberately contains no graph discovery, causal
//! inference, intervention search, or real-world institutional classification.
//! It only defines how finding state may change without laundering authority.

mod finding_state;

pub use finding_state::*;
