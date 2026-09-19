// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// Compile the pre-registration keeper storage modules directly under
// `--features studio`. Their internal unit tests then execute without claiming
// the large `muse_studio` binary has been wired yet.
#[path = "../src/keeper_semantic_store.rs"]
mod keeper_semantic_store;
#[path = "../src/keeper_score_store.rs"]
mod keeper_score_store;
