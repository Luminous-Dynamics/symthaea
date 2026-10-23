// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed claim-admissibility contract for psych-bench results.
//!
//! A numeric benchmark result is not, by itself, authority for every claim a
//! downstream report could attach to it. This module makes that distinction
//! machine-readable without changing the historical `BenchmarkResult` schema in
//! place. Consumers that persist or publish benchmark evidence should wrap the
//! raw result in [`ClaimedBenchmarkResult`] (or use [`run_claimed`]).
//!
//! The contract is intentionally conservative:
//!
//! - ordinary psychometric tasks default to [`BenchmarkClaimClass::BehavioralMeasurement`];
//! - the known shared-mask/toy-crypto benchmarks are identified by concrete Rust
//!   type and are always [`BenchmarkClaimClass::InsecureAlgebraDemonstration`];
//! - only [`BenchmarkClaimClass::SecurityPropertyTest`] can authorize a positive
//!   security-property claim;
//! - scientific validation, provenance, high scores, and display names cannot
//!   implicitly create security authority.
//!
//! This is the first migration tranche for issue #1164. The legacy
//! `BenchmarkResult` / `BenchmarkReport` / snapshot types remain readable while
//! report and snapshot producers migrate to this explicit envelope.

use crate::benchmarks::security::{
    CollectiveAggregationBenchmark, CrossMaskPrivacyBenchmark, EncryptedBindingBenchmark,
    EncryptedClassificationBenchmark, EncryptedLearningBenchmark, ScalingAnalysisBenchmark,
};
use crate::harness::{BenchmarkConfig, BenchmarkResult, PsychBenchmark};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::collections::BTreeMap;

/// What kind of claim a benchmark is designed to support.
///
/// These variants are categories, not a universal strength ordering. In
/// particular, `ScientificValidation` does not imply `SecurityPropertyTest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkClaimClass {
    /// Behavioral/psychometric measurement without mechanism or security authority.
    BehavioralMeasurement,
    /// Demonstrates behavior of a mechanism, without establishing an external property.
    MechanismDemonstration,
    /// Demonstrates algebra/fidelity of a construction known not to provide the
    /// cryptographic property suggested by historical terminology.
    InsecureAlgebraDemonstration,
    /// A deliberately vulnerable/adversarial security control. Negative evidence only.
    SecurityNegativeControl,
    /// A benchmark whose design is explicitly admissible as evidence for a named
    /// security property. This is the only positive security-authorizing class.
    SecurityPropertyTest,
    /// Scientific validation against an external/independent empirical target.
    ScientificValidation,
}

/// Security authority carried by a claim class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityAuthority {
    /// No cryptographic/security claim may be inferred from the result.
    None,
    /// May establish a vulnerability / negative-control result, never a positive property.
    NegativeOnly,
    /// May contribute evidence for the explicitly tested positive security property.
    PositivePropertyEvidence,
}

impl BenchmarkClaimClass {
    /// Security authority is derived from claim class, not from scores, citations,
    /// benchmark names, or generic "scientific" status.
    pub const fn security_authority(self) -> SecurityAuthority {
        match self {
            Self::SecurityNegativeControl => SecurityAuthority::NegativeOnly,
            Self::SecurityPropertyTest => SecurityAuthority::PositivePropertyEvidence,
            Self::BehavioralMeasurement
            | Self::MechanismDemonstration
            | Self::InsecureAlgebraDemonstration
            | Self::ScientificValidation => SecurityAuthority::None,
        }
    }

    /// Whether this class is allowed to support a positive security-property claim.
    pub const fn supports_positive_security_claim(self) -> bool {
        matches!(
            self.security_authority(),
            SecurityAuthority::PositivePropertyEvidence
        )
    }
}

/// Return the claim class for a concrete benchmark type.
///
/// The insecure shared-mask suite is matched by concrete Rust `TypeId`, not its
/// human-readable `name()`. Renaming `InsecureAlgebraDemo::...` therefore cannot
/// accidentally change claim semantics.
///
/// New psychometric benchmarks conservatively default to behavioral measurement.
/// A benchmark seeking stronger mechanism/scientific/security authority must be
/// added to this reviewed contract rather than acquiring it from numeric output.
pub fn claim_class_for<B: PsychBenchmark + 'static>(_: &B) -> BenchmarkClaimClass {
    let type_id = TypeId::of::<B>();

    if type_id == TypeId::of::<CollectiveAggregationBenchmark>()
        || type_id == TypeId::of::<CrossMaskPrivacyBenchmark>()
        || type_id == TypeId::of::<EncryptedBindingBenchmark>()
        || type_id == TypeId::of::<EncryptedClassificationBenchmark>()
        || type_id == TypeId::of::<EncryptedLearningBenchmark>()
        || type_id == TypeId::of::<ScalingAnalysisBenchmark>()
    {
        BenchmarkClaimClass::InsecureAlgebraDemonstration
    } else {
        BenchmarkClaimClass::BehavioralMeasurement
    }
}

/// A benchmark result with explicit, serialized claim admissibility.
///
/// `claim_class` intentionally has no serde default. A persisted envelope that
/// omits it is malformed rather than silently inheriting stronger semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimedBenchmarkResult {
    pub claim_class: BenchmarkClaimClass,
    pub result: BenchmarkResult,
}

impl ClaimedBenchmarkResult {
    /// Attach the reviewed claim class for `benchmark` to an already-computed result.
    pub fn attach<B: PsychBenchmark + 'static>(benchmark: &B, result: BenchmarkResult) -> Self {
        Self {
            claim_class: claim_class_for(benchmark),
            result,
        }
    }

    /// Security authority derived solely from the typed claim class.
    pub const fn security_authority(&self) -> SecurityAuthority {
        self.claim_class.security_authority()
    }

    /// Positive security authority is intentionally impossible for insecure demos,
    /// even when every numeric metric is perfect.
    pub const fn supports_positive_security_claim(&self) -> bool {
        self.claim_class.supports_positive_security_claim()
    }
}

/// Execute a benchmark and immediately bind the raw result to its claim class.
pub fn run_claimed<B: PsychBenchmark + 'static>(
    benchmark: &B,
    config: &BenchmarkConfig,
) -> ClaimedBenchmarkResult {
    ClaimedBenchmarkResult::attach(benchmark, benchmark.run(config))
}

/// Error returned when a caller tries to collapse heterogeneous claim classes
/// into one undifferentiated class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimAggregationError {
    Empty,
    MixedClaimClasses(BTreeMap<BenchmarkClaimClass, usize>),
}

/// Partitioned benchmark report that preserves claim classes.
///
/// There is deliberately no "overall claim score" method. Consumers can inspect
/// partitions or require one uniform class; heterogeneous evidence never silently
/// becomes a stronger scalar authority.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClaimedBenchmarkReport {
    pub results: Vec<ClaimedBenchmarkResult>,
}

impl ClaimedBenchmarkReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, result: ClaimedBenchmarkResult) {
        self.results.push(result);
    }

    pub fn run_and_add<B: PsychBenchmark + 'static>(
        &mut self,
        benchmark: &B,
        config: &BenchmarkConfig,
    ) {
        self.add(run_claimed(benchmark, config));
    }

    /// Count results by claim class so mixed evidence stays visibly partitioned.
    pub fn claim_partition(&self) -> BTreeMap<BenchmarkClaimClass, usize> {
        let mut partition = BTreeMap::new();
        for result in &self.results {
            *partition.entry(result.claim_class).or_insert(0) += 1;
        }
        partition
    }

    /// Require a single claim class. Mixed classes fail closed instead of being
    /// collapsed into one inferred authority level.
    pub fn require_uniform_claim_class(
        &self,
    ) -> Result<BenchmarkClaimClass, ClaimAggregationError> {
        let partition = self.claim_partition();
        match partition.len() {
            0 => Err(ClaimAggregationError::Empty),
            1 => Ok(*partition.keys().next().expect("one entry")),
            _ => Err(ClaimAggregationError::MixedClaimClasses(partition)),
        }
    }

    /// Results admissible as positive security-property evidence.
    ///
    /// Insecure algebra demonstrations and generic scientific validations are
    /// excluded regardless of their numeric metrics.
    pub fn positive_security_property_results(&self) -> Vec<&ClaimedBenchmarkResult> {
        self.results
            .iter()
            .filter(|result| result.supports_positive_security_claim())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::worm::NBackBenchmark;
    use crate::harness::BenchmarkProvenance;

    fn assert_insecure_demo<B: PsychBenchmark + 'static>(benchmark: &B) {
        assert_eq!(
            claim_class_for(benchmark),
            BenchmarkClaimClass::InsecureAlgebraDemonstration
        );
        assert_eq!(
            claim_class_for(benchmark).security_authority(),
            SecurityAuthority::None
        );
        assert!(!claim_class_for(benchmark).supports_positive_security_claim());
    }

    #[test]
    fn all_shared_mask_benchmarks_are_insecure_algebra_demonstrations() {
        assert_insecure_demo(&CollectiveAggregationBenchmark);
        assert_insecure_demo(&CrossMaskPrivacyBenchmark);
        assert_insecure_demo(&EncryptedBindingBenchmark);
        assert_insecure_demo(&EncryptedClassificationBenchmark);
        assert_insecure_demo(&EncryptedLearningBenchmark);
        assert_insecure_demo(&ScalingAnalysisBenchmark);
    }

    #[test]
    fn ordinary_psychometric_benchmark_defaults_to_behavioral_measurement() {
        assert_eq!(
            claim_class_for(&NBackBenchmark),
            BenchmarkClaimClass::BehavioralMeasurement
        );
        assert_eq!(
            claim_class_for(&NBackBenchmark).security_authority(),
            SecurityAuthority::None
        );
    }

    #[test]
    fn positive_security_authority_is_explicit_not_hierarchical() {
        assert!(BenchmarkClaimClass::SecurityPropertyTest.supports_positive_security_claim());
        assert!(!BenchmarkClaimClass::ScientificValidation.supports_positive_security_claim());
        assert!(!BenchmarkClaimClass::MechanismDemonstration.supports_positive_security_claim());
        assert!(!BenchmarkClaimClass::InsecureAlgebraDemonstration.supports_positive_security_claim());
    }

    #[test]
    fn display_name_changes_do_not_change_attached_claim_semantics() {
        let raw = BenchmarkResult::new("arbitrary-display-name", None);
        let mut claimed = ClaimedBenchmarkResult::attach(&CollectiveAggregationBenchmark, raw);
        assert_eq!(
            claimed.claim_class,
            BenchmarkClaimClass::InsecureAlgebraDemonstration
        );

        claimed.result.benchmark = "renamed-without-insecure-prefix".to_string();
        assert_eq!(
            claimed.claim_class,
            BenchmarkClaimClass::InsecureAlgebraDemonstration
        );
        assert_eq!(claimed.security_authority(), SecurityAuthority::None);
    }

    #[test]
    fn perfect_numeric_results_cannot_elevate_insecure_demo_authority() {
        let mut raw = BenchmarkResult::new("perfect-demo", None);
        raw.insert(
            "fidelity",
            crate::harness::MetricValue {
                mean: 1.0,
                std_dev: 0.0,
                n: 1000,
                ci_lower: 1.0,
                ci_upper: 1.0,
            },
        );
        let claimed = ClaimedBenchmarkResult::attach(&EncryptedClassificationBenchmark, raw);
        assert_eq!(
            claimed.claim_class,
            BenchmarkClaimClass::InsecureAlgebraDemonstration
        );
        assert!(!claimed.supports_positive_security_claim());
    }

    #[test]
    fn claimed_result_serialization_preserves_class_and_omission_fails_closed() {
        let raw = BenchmarkResult::new("serialization-test", None);
        let claimed = ClaimedBenchmarkResult::attach(&CollectiveAggregationBenchmark, raw);

        let json = serde_json::to_string(&claimed).expect("serialize claimed benchmark result");
        let round_trip: ClaimedBenchmarkResult =
            serde_json::from_str(&json).expect("deserialize claimed benchmark result");
        assert_eq!(round_trip.claim_class, claimed.claim_class);

        let mut value = serde_json::to_value(&claimed).expect("serialize to value");
        value
            .as_object_mut()
            .expect("claimed result is an object")
            .remove("claim_class");
        assert!(
            serde_json::from_value::<ClaimedBenchmarkResult>(value).is_err(),
            "missing claim_class must not silently deserialize"
        );
    }

    #[test]
    fn mixed_claim_classes_cannot_collapse_to_uniform_authority() {
        let insecure = ClaimedBenchmarkResult::attach(
            &CollectiveAggregationBenchmark,
            BenchmarkResult::new("insecure", None),
        );
        let behavioral = ClaimedBenchmarkResult::attach(
            &NBackBenchmark,
            BenchmarkResult::new("behavioral", None),
        );

        let mut report = ClaimedBenchmarkReport::new();
        report.add(insecure);
        report.add(behavioral);

        let err = report
            .require_uniform_claim_class()
            .expect_err("mixed claim classes must fail closed");
        match err {
            ClaimAggregationError::MixedClaimClasses(partition) => {
                assert_eq!(
                    partition.get(&BenchmarkClaimClass::InsecureAlgebraDemonstration),
                    Some(&1)
                );
                assert_eq!(
                    partition.get(&BenchmarkClaimClass::BehavioralMeasurement),
                    Some(&1)
                );
            }
            other => panic!("unexpected aggregation error: {other:?}"),
        }
        assert!(report.positive_security_property_results().is_empty());
    }

    #[test]
    fn provenance_does_not_elevate_claim_class() {
        struct CitedBehavioral;
        impl PsychBenchmark for CitedBehavioral {
            fn name(&self) -> &str {
                "CitedBehavioral"
            }

            fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
                BenchmarkResult::new(self.name(), None)
            }

            fn provenance(&self) -> Option<BenchmarkProvenance> {
                Some(BenchmarkProvenance {
                    paradigm: "Cited test",
                    citation: "Example (2026)",
                    year: 2026,
                    doi: Some("10.0000/example"),
                })
            }
        }

        let benchmark = CitedBehavioral;
        assert!(benchmark.provenance().is_some());
        assert_eq!(
            claim_class_for(&benchmark),
            BenchmarkClaimClass::BehavioralMeasurement
        );
        assert_eq!(
            claim_class_for(&benchmark).security_authority(),
            SecurityAuthority::None
        );
    }
}
