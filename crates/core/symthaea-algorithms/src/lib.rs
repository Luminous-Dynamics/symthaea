// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-first algorithm registry and discovery contracts for Symthaea.
//!
//! This crate is descriptive, not authorizing:
//! `Problem != Algorithm != Implementation != Evaluation != Evidence != Promotion != Authority`.

pub mod discovery;
pub mod evaluation;
pub mod pareto;
pub mod replication;
mod registry;

pub use registry::*;
