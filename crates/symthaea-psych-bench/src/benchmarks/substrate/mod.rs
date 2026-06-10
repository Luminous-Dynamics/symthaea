// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate independence domain benchmarks.
//!
//! - **Transfer** — Substrate Transfer fidelity (Multiple Realizability, Putnam 1967)
//! - **Degradation** — Graceful degradation under substrate quality loss (Tononi, 2004)
//! - **Latency** — Processing speed impact across substrate types (Koch et al., 2016)

pub mod degradation;
pub mod latency;
pub mod transfer;
pub mod validation;

pub use degradation::SubstrateDegradationBenchmark;
pub use latency::SubstrateLatencyBenchmark;
pub use transfer::SubstrateTransferBenchmark;
pub use validation::SubstrateValidationBenchmark;
