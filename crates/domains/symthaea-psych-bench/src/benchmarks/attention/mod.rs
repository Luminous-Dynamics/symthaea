// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Attention domain benchmarks.
//!
//! - **AttentionalBlink** — Temporal attention limits in RSVP streams
//! - **VisualSearch** — Parallel vs serial attentional processing
//! - **MismatchNegativity** — Pre-attentive oddball detection

pub mod attentional_blink;
pub mod mismatch_negativity;
pub mod visual_search;

pub use attentional_blink::AttentionalBlinkBenchmark;
pub use mismatch_negativity::MismatchNegativityBenchmark;
pub use visual_search::VisualSearchBenchmark;
