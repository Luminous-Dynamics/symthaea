// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ToMBench: Theory of Mind benchmark suite.
//!
//! Tests `SocialCoherence` module with mental model tracking,
//! belief/desire/intention attribution via 5 task types.
//!
//! When the `symthaea-backend` feature is enabled, each benchmark uses
//! the FEP `ActiveInferenceAgent` to predict *behavior* from mental states
//! (Applied ToM), proving behavioral reasoning that LLMs lack.

#[cfg(feature = "symthaea-backend")]
pub mod applied_tom;
pub mod false_belief;
pub mod faux_pas;
pub mod hinting;
pub mod persuasion;
pub mod strange_story;

pub use false_belief::FalseBeliefBenchmark;
pub use faux_pas::FauxPasBenchmark;
pub use hinting::HintingBenchmark;
pub use persuasion::PersuasionBenchmark;
pub use strange_story::StrangeStoryBenchmark;
