// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Commons Shared Types & Utilities
//!
//! Common functionality for all domain zomes in the Commons cluster:
//! - Batch query operations (solving N+1 query patterns)
//! - Anchor utilities for consistent indexing
//! - Bridge types for cross-domain communication
//! - Geographic types shared across domains

pub mod anchors;
pub mod batch;
pub mod bridge_types;
pub mod geo;

pub use anchors::*;
pub use batch::*;
pub use bridge_types::*;
pub use geo::*;
