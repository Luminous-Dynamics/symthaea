// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gold nanojunction physical-reservoir research models for Symthaea.
//!
//! The effective network model and the canonical evidence-binding serializer
//! are kept in separate modules so physical dynamics and experiment identity
//! cannot silently drift together.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod config_binding;
mod model;

pub use model::*;
