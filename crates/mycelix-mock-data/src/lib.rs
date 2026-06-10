// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared mock data for all Mycelix cluster frontends.
//!
//! Provides consistent, realistic mock data for offline-first rendering.
//! Each cluster's `context.rs` imports from here as the graceful
//! degradation fallback when the Holochain conductor is unavailable.

pub mod climate;
pub mod energy;
pub mod knowledge;
pub mod finance;
