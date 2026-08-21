// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-way bridge from Symthaea's internal `StructuredThought` IR into SCIP.
//!
//! Semantic data and trusted renderer control are kept separate. Native Broca
//! SSM/L-SSM generation does not use this crate and retains its direct
//! ThoughtChannels path.

#![forbid(unsafe_code)]

mod adapter;

pub use adapter::*;
