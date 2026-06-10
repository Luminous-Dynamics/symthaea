// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inhibition domain benchmarks.
//!
//! - **Go/No-Go** — Prepotent response inhibition (commission errors)
//! - **Stop Signal** — Inhibitory control latency (SSRT)
//! - **FlankerInhibition** — Interference suppression (Eriksen & Eriksen, 1974)

pub mod flanker_inhibition;
pub mod go_nogo;
pub mod stop_signal;

pub use flanker_inhibition::FlankerInhibitionBenchmark;
pub use go_nogo::GoNoGoBenchmark;
pub use stop_signal::StopSignalBenchmark;
