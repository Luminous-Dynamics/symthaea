// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Local shim: re-exports serde as serde_core for bitflags 2.10+ compatibility
// with holochain's pinned serde =1.0.219.
pub use serde::*;
