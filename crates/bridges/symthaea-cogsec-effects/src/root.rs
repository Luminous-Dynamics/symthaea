// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public root for canonical CogSec effect and resource-state commitments.
//!
//! Effect identity and resource-state identity are kept in one dependency-neutral
//! bridge so the trusted evaluation adapter and post-legacy observer can share the
//! exact same canonical representations without moving hashing into the logical
//! reference-monitor core.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[path = "lib.rs"]
mod effects;
pub use effects::*;

mod state_commitments;
pub use state_commitments::*;
