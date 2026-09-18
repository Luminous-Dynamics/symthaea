// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active grounded-dream implementation for SYM-RSI-001.
//!
//! v6 is preserved separately as `sym_rsi_grounded_dream_v6.rs`. The active public
//! surface re-exports the preregistered v7 structural-support implementation.

#[path = "sym_rsi_grounded_dream_v7.rs"]
mod v7;

pub use v7::*;
