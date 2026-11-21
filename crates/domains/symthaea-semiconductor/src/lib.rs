// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Rust-native semiconductor engineering contracts.
//!
//! This crate is separate from the general additive-fabrication kernel. V1
//! starts with benchmark/source identity and split-manifest validation; it
//! does not define process recipes, equipment control, or fabrication authority.

pub mod benchmark;
