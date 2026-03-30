// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Re-export all data types from terra-atlas-core.
//! This module exists so that existing `use crate::data::types::*` imports
//! continue to work unchanged throughout the codebase.

pub use terra_atlas_core::types::*;
