// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Electro-acoustic engineering models for Symthaea.
//!
//! The crate is split so the EAC-001 provenance-bound parameter model remains
//! independently inspectable while later analytical models build on top of it.
//! Physical observation authority remains outside this crate in FIELD.

#![deny(unsafe_code)]

#[path = "lib.rs"]
mod parameters;

pub use parameters::*;

pub mod enclosure;
pub mod linear;
