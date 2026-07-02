// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pagination types for bounded query results.
//!
//! Re-exports the canonical pagination primitives from `mycelix-bridge-common`
//! so that domain zomes within the Commons cluster can import them from a
//! single, familiar crate.

pub use mycelix_bridge_common::{PaginationInput, PaginatedResponse, paginate_links};
