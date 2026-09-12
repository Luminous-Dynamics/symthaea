// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crate entry for verified welfare-intervention authority.

#![deny(unsafe_code)]

#[path = "lib.rs"]
#[allow(unused_imports)]
mod implementation;

pub use implementation::*;
