// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Direct qualification harness for the pre-wiring keeper semantic store.
//!
//! The production `muse_studio` binary wires this module in the following
//! tranche. Keeping this harness path-based lets the storage boundary execute
//! independently first without pretending it already has live server authority.

#[path = "../src/keeper_semantic_store.rs"]
mod keeper_semantic_store;
