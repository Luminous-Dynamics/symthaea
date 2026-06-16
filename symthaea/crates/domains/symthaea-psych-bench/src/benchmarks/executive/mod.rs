// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Executive function benchmarks.
//!
//! - **WCST** — Wisconsin Card Sorting Test (rule learning, set-shifting, perseveration)
//! - **IGT** — Iowa Gambling Task (decision-making under ambiguity, loss aversion)
//! - **Raven's** — Progressive Matrices (fluid intelligence, pattern completion)
//! - **Stroop** — Color-Word Interference (inhibitory control, selective attention)
//! - **Flanker** — Eriksen Flanker Task (spatial attention filtering, response inhibition)
//! - **Tower of London** — Multi-step planning (executive planning, lookahead depth)
//! - **Dual-Task** — Concurrent memory load + choice RT (resource sharing)

pub mod dual_task;
pub mod flanker;
pub mod iowa_gambling;
pub mod ravens;
pub mod stroop;
pub mod tower_of_london;
pub mod wisconsin;

pub use dual_task::DualTaskBenchmark;
pub use flanker::FlankerBenchmark;
pub use iowa_gambling::IowaGamblingBenchmark;
pub use ravens::RavensProgressiveMatricesBenchmark;
pub use stroop::StroopBenchmark;
pub use tower_of_london::TowerOfLondonBenchmark;
pub use wisconsin::WisconsinCardSortingBenchmark;
