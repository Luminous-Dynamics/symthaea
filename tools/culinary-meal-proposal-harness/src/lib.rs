// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Focused harness for the exact production culinary meal proposal source.
//!
//! This module contains its own typed contract and in-source tests. This harness
//! does not resolve or compile the complete Symthaea workspace.

#[path = "../../../crates/domains/symthaea-culinary/src/meal_proposal.rs"]
pub mod meal_proposal;
