// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language processing domain benchmarks.
//!
//! - **GardenPath** — Garden-path sentence processing (disambiguation cost)
//! - **SemanticCoherence** — Topic coherence, drift, and recovery in generated text
//! - **LexicalDecision** — Word vs non-word discrimination (lexicality + frequency effects)
//! - **SemanticPriming** — Related words speed processing (SOA modulation)

pub mod garden_path;
pub mod lexical_decision;
pub mod semantic_coherence;
pub mod semantic_priming;

pub use garden_path::GardenPathBenchmark;
pub use lexical_decision::LexicalDecisionBenchmark;
pub use semantic_coherence::SemanticCoherenceBenchmark;
pub use semantic_priming::SemanticPrimingBenchmark;
