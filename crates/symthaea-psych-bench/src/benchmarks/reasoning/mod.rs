// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fluid reasoning benchmarks.

pub mod arc_abductive;
pub mod arc_algebra;
pub mod arc_analogy;
pub mod arc_chain;
pub mod arc_compositional;
pub mod arc_dataset;
pub mod arc_fewshot;
pub mod arc_fluid;
pub mod arc_noise;
pub mod arc_rsa;
pub mod arc_scaling;
pub mod arc_staircase;

pub use arc_abductive::ArcAbductiveBenchmark;
pub use arc_algebra::ArcAlgebraBenchmark;
pub use arc_analogy::ArcAnalogyBenchmark;
pub use arc_chain::ArcChainBenchmark;
pub use arc_compositional::ArcCompositionalBenchmark;
pub use arc_fewshot::ArcFewShotBenchmark;
pub use arc_fluid::ArcFluidBenchmark;
pub use arc_noise::ArcNoiseBenchmark;
pub use arc_rsa::ArcRsaBenchmark;
pub use arc_scaling::ArcScalingBenchmark;
pub use arc_staircase::ArcStaircaseBenchmark;
