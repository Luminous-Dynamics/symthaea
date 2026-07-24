// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//!
//! Quarantined shared-mask algebra benchmarks. These measure functional
//! fidelity and leakage in an insecure toy transform; they are not security,
//! FHE, privacy, or threshold-cryptography benchmarks.
pub mod collective_aggregation;
pub mod cross_mask_privacy;
pub mod encrypted_binding;
pub mod encrypted_classification;
pub mod encrypted_learning;
pub mod scaling_analysis;

pub use collective_aggregation::CollectiveAggregationBenchmark;
pub use cross_mask_privacy::CrossMaskPrivacyBenchmark;
pub use encrypted_binding::EncryptedBindingBenchmark;
pub use encrypted_classification::EncryptedClassificationBenchmark;
pub use encrypted_learning::EncryptedLearningBenchmark;
pub use scaling_analysis::ScalingAnalysisBenchmark;
