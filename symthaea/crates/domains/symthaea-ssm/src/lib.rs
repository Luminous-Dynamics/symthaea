// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea SSM - diagonal state space model core for edge inference.

pub mod selective_scan;

pub use selective_scan::{SelectiveParams, SsmState};
