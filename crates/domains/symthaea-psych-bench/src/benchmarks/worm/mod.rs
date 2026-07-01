// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WorM: Working Memory benchmark suite.
//!
//! Tests `ContinuousMind` with 7+/-2 capacity, FIFO eviction, activation decay
//! against published human baselines for working memory tasks.

pub mod binding;
pub mod change_detection;
pub mod digit_span;
pub mod nback;
pub mod serial_recall;
pub mod spatial_updating;

pub use binding::BindingBenchmark;
pub use change_detection::ChangeDetectionBenchmark;
pub use digit_span::DigitSpanBenchmark;
pub use nback::NBackBenchmark;
pub use serial_recall::SerialRecallBenchmark;
pub use spatial_updating::SpatialUpdatingBenchmark;
