// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vocal tract quality metrics — re-exported from `symthaea-vocal-tract` sub-crate.
//!
//! See `crates/symthaea-vocal-tract/src/metrics.rs` for the canonical implementation.

#[cfg(feature = "vocal-tract")]
pub use symthaea_vocal_tract::metrics::{VocalTractMetrics, load_wav, save_wav};
