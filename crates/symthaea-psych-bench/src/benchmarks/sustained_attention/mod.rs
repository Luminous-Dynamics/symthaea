// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sustained Attention domain benchmarks.
//!
//! - **SART** — Sustained Attention to Response Task (commission errors)
//! - **PVT** — Psychomotor Vigilance Task (fatigue-related RT slowing)
//! - **CPT** — Continuous Performance Task (2-back sequential matching)

pub mod cpt;
pub mod pvt;
pub mod sart;

pub use cpt::CptBenchmark;
pub use pvt::PvtBenchmark;
pub use sart::SartBenchmark;
