// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query System
//!
//! Provides efficient searching and filtering of epistemic claims with indexing

pub mod filter;
pub mod index;
pub mod engine;

pub use engine::QueryEngine;
pub use filter::{QueryFilter, SortBy, SortOrder};
pub use index::ClaimIndex;
