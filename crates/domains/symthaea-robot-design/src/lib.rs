// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bounded robot-design semantics.
//!
//! The qualified ROB-DESIGN-001A implementation is preserved byte-for-byte in
//! the private `canonical` module and re-exported at this crate root. Staged
//! follow-on semantics live in focused sibling modules.

#![forbid(unsafe_code)]

mod canonical;
pub mod exact_parameters;

pub use canonical::*;
