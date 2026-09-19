// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

// Compile the pre-registration persisted-fingerprint admission boundary directly
// so its fail-closed tests run before the Atlas monolith is changed.
#[path = "../src/keeper_fingerprint_evidence.rs"]
mod keeper_fingerprint_evidence;
