// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing mathematical research primitives for Symthaea.
//!
//! Authority is split by module so specification, claims, verification, and
//! later novelty/research-policy layers cannot silently collapse into one
//! confidence score.

pub mod spec;

pub use spec::*;
