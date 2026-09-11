// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public facade for the lunar-launcher research crate.
//!
//! The LL-001/002 implementation remains in `lib.rs` and is re-exported intact;
//! LL-003B forward ballistic dynamics lives in a separate module so trajectory
//! work cannot quietly rewrite the launcher/catcher vocabulary.

#![deny(unsafe_code)]

#[path = "lib.rs"]
mod ll001_002;

pub use ll001_002::*;

pub mod ballistics;
