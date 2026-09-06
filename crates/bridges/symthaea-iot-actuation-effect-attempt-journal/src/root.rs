// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![deny(unsafe_code)]

#[path = "lib.rs"]
mod local;
pub use local::*;

mod protected;
pub use protected::*;
