// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// Compile the pre-registration production modules directly so their unit tests
// execute under `--features studio` before the tiny `muse_studio` wiring tranche.
#[path = "../src/keeper_semantic_store.rs"]
mod keeper_semantic_store;
#[path = "../src/keeper_semantic_boundary.rs"]
mod keeper_semantic_boundary;
