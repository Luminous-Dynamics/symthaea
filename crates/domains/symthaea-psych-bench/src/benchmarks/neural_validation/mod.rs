// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Validation domain — comparing Symthaea dynamics against brain encoding models.
//!
//! 8 benchmarks spanning fMRI, EEG, temporal dynamics, parcellation robustness,
//! bidirectional validation, substrate comparison, evidence upgrade, and hybrid optimization.

pub mod cortical_similarity;

pub use cortical_similarity::BidirectionalValidationBenchmark;
pub use cortical_similarity::CorticalSimilarityBenchmark;
pub use cortical_similarity::EegValidationBenchmark;
pub use cortical_similarity::EvidenceUpgradeBenchmark;
pub use cortical_similarity::HybridSubstrateBenchmark;
pub use cortical_similarity::ParcellationRobustnessBenchmark;
pub use cortical_similarity::SubstrateComparisonBenchmark;
pub use cortical_similarity::TemporalDynamicsBenchmark;
