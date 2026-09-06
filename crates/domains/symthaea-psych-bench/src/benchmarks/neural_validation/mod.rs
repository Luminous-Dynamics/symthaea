// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural-evidence research qualification boundary.
//!
//! The historical neural-validation experiments are preserved for repair and
//! model-behavior regression work, but none is currently qualified as public
//! external-neural evidence.
//!
//! The quarantine is intentional. The legacy experiments contain one or more
//! of the following scientific-boundary defects:
//!
//! - implicit synthetic fallback when qualified external data are absent,
//! - hand-authored pseudo-TRIBE predictions,
//! - synthetic substrate profiles whose expected ranking is built into the test,
//! - atlas labels that do not actually change the underlying parcellation/data,
//! - EEG values deterministically derived from the same regional inputs on both sides,
//! - legacy 12-region TRIBE ingestion without a qualified native-surface transform,
//! - evidence-upgrade semantics that are forbidden by the substrate evidence boundary.
//!
//! Until provenance-aware ingestion and reviewed coordinate transforms exist,
//! these experiments must not be imported as validated fMRI/EEG/substrate claims.
//! See `docs/neuroscience/NEURAL_BENCHMARK_QUALIFICATION_V1.md`.

// Kept crate-private so the code can continue to compile, be tested, and be
// repaired without presenting the historical experiments as a public scientific
// validation API. Some formerly public experiment structs are intentionally
// unreachable outside this crate during quarantine, so dead-code allowance is
// scoped to this legacy module rather than weakening linting globally.
#[allow(dead_code)]
pub(crate) mod cortical_similarity;

/// Qualification state of the public external-neural benchmark surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralValidationQualification {
    /// No historical neural benchmark currently satisfies the external-evidence
    /// provenance, coordinate, independence, and claim-authority requirements.
    Quarantined,
}

/// Current qualification state for this benchmark surface.
pub const QUALIFICATION: NeuralValidationQualification = NeuralValidationQualification::Quarantined;

/// Publicly qualified external-neural benchmarks.
///
/// This list is deliberately empty until an experiment consumes admitted,
/// provenance-bearing external or empirical neural observations through a
/// reviewed coordinate pipeline.
pub const QUALIFIED_EXTERNAL_BENCHMARKS: &[&str] = &[];

/// Historical experiments retained behind the quarantine boundary.
pub const QUARANTINED_BENCHMARKS: &[&str] = &[
    "CorticalSimilarity",
    "TemporalDynamics",
    "BidirectionalValidation",
    "SubstrateComparison",
    "ParcellationRobustness",
    "EvidenceUpgrade",
    "EegValidation",
    "HybridSubstrate",
];

/// Stable human-readable reason for the quarantine.
pub const QUALIFICATION_REASON: &str =
    "no historical neural benchmark currently has qualified external evidence + reviewed coordinate lineage + independent claim authority";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_external_neural_benchmark_is_publicly_qualified() {
        assert_eq!(QUALIFICATION, NeuralValidationQualification::Quarantined);
        assert!(QUALIFIED_EXTERNAL_BENCHMARKS.is_empty());
    }

    #[test]
    fn all_eight_historical_benchmarks_are_accounted_for() {
        assert_eq!(QUARANTINED_BENCHMARKS.len(), 8);
        for required in [
            "CorticalSimilarity",
            "TemporalDynamics",
            "BidirectionalValidation",
            "SubstrateComparison",
            "ParcellationRobustness",
            "EvidenceUpgrade",
            "EegValidation",
            "HybridSubstrate",
        ] {
            assert!(QUARANTINED_BENCHMARKS.contains(&required));
        }
    }

    #[test]
    fn qualification_reason_is_explicit() {
        assert!(QUALIFICATION_REASON.contains("qualified external evidence"));
        assert!(QUALIFICATION_REASON.contains("coordinate lineage"));
        assert!(QUALIFICATION_REASON.contains("claim authority"));
    }
}
