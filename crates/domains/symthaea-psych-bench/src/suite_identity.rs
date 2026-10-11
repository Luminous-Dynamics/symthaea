// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content-addressed identity for the executable benchmark population of a run.
//!
//! This module answers one narrow question: which [`PsychBenchmark`] objects
//! were in the executable suite? It does not establish that the suite is
//! scientifically valid, complete for a paper, comparable to another task set,
//! or successfully executed.

use crate::harness::PsychBenchmark;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const BENCHMARK_SUITE_IDENTITY_SCHEMA_VERSION: &str =
    "psych-benchmark-suite-executable-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "label")]
pub enum BenchmarkSuitePurpose {
    FullBattery,
    PaperBattery,
    RegressionBattery,
    MultiSeedRobustness,
    ReliabilitySubset,
    SatSubset,
    Custom(String),
}

impl BenchmarkSuitePurpose {
    fn canonical_label(&self) -> Result<String, BenchmarkSuiteIdentityError> {
        let label = match self {
            Self::FullBattery => "full_battery".to_string(),
            Self::PaperBattery => "paper_battery".to_string(),
            Self::RegressionBattery => "regression_battery".to_string(),
            Self::MultiSeedRobustness => "multi_seed_robustness".to_string(),
            Self::ReliabilitySubset => "reliability_subset".to_string(),
            Self::SatSubset => "sat_subset".to_string(),
            Self::Custom(label) => {
                validate_component("suite purpose", label)?;
                format!("custom:{label}")
            }
        };
        Ok(label)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutableBenchmarkSuiteIdentity {
    pub schema_version: String,
    pub purpose: BenchmarkSuitePurpose,
    /// Canonical lexicographic order. Runtime registration order is not semantic.
    pub benchmark_ids: Vec<String>,
    pub benchmark_count: usize,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkSuiteIdentityError {
    EmptySuite,
    EmptyBenchmarkId,
    NonCanonicalBenchmarkId { benchmark_id: String },
    DuplicateBenchmarkId { benchmark_id: String },
    InvalidComponent { field: &'static str, value: String },
    NonCanonicalOrdering,
    BenchmarkCountMismatch { declared: usize, actual: usize },
    DigestMismatch,
}

impl ExecutableBenchmarkSuiteIdentity {
    /// Build identity from the actual executable benchmark trait objects.
    ///
    /// The caller chooses a purpose, but membership comes from the runnable
    /// objects themselves. Registration order is deliberately non-semantic.
    pub fn from_benchmarks<'a>(
        purpose: BenchmarkSuitePurpose,
        benchmarks: impl IntoIterator<Item = &'a dyn PsychBenchmark>,
    ) -> Result<Self, BenchmarkSuiteIdentityError> {
        Self::from_ids(
            purpose,
            benchmarks.into_iter().map(|benchmark| benchmark.name().to_string()),
        )
    }

    /// Build identity from benchmark IDs when the executable objects were
    /// already resolved by another layer.
    ///
    /// Prefer [`Self::from_benchmarks`] at execution boundaries. This helper is
    /// useful for receipt replay and independent canonicalization tests; it does
    /// not itself prove that arbitrary supplied strings resolve to implementations.
    pub fn from_ids(
        purpose: BenchmarkSuitePurpose,
        benchmark_ids: impl IntoIterator<Item = String>,
    ) -> Result<Self, BenchmarkSuiteIdentityError> {
        let _ = purpose.canonical_label()?;

        let mut seen = BTreeSet::new();
        for benchmark_id in benchmark_ids {
            validate_benchmark_id(&benchmark_id)?;
            if !seen.insert(benchmark_id.clone()) {
                return Err(BenchmarkSuiteIdentityError::DuplicateBenchmarkId {
                    benchmark_id,
                });
            }
        }
        if seen.is_empty() {
            return Err(BenchmarkSuiteIdentityError::EmptySuite);
        }

        let benchmark_ids = seen.into_iter().collect::<Vec<_>>();
        let benchmark_count = benchmark_ids.len();
        let digest = compute_digest(&purpose, &benchmark_ids)?;
        Ok(Self {
            schema_version: BENCHMARK_SUITE_IDENTITY_SCHEMA_VERSION.to_string(),
            purpose,
            benchmark_ids,
            benchmark_count,
            digest,
        })
    }

    pub fn validate(&self) -> Result<(), BenchmarkSuiteIdentityError> {
        if self.schema_version != BENCHMARK_SUITE_IDENTITY_SCHEMA_VERSION {
            return Err(BenchmarkSuiteIdentityError::InvalidComponent {
                field: "schema_version",
                value: self.schema_version.clone(),
            });
        }
        let _ = self.purpose.canonical_label()?;
        if self.benchmark_ids.is_empty() {
            return Err(BenchmarkSuiteIdentityError::EmptySuite);
        }
        for benchmark_id in &self.benchmark_ids {
            validate_benchmark_id(benchmark_id)?;
        }
        if self
            .benchmark_ids
            .windows(2)
            .any(|window| window[0] >= window[1])
        {
            return Err(BenchmarkSuiteIdentityError::NonCanonicalOrdering);
        }
        if self.benchmark_count != self.benchmark_ids.len() {
            return Err(BenchmarkSuiteIdentityError::BenchmarkCountMismatch {
                declared: self.benchmark_count,
                actual: self.benchmark_ids.len(),
            });
        }
        let expected = compute_digest(&self.purpose, &self.benchmark_ids)?;
        if self.digest != expected {
            return Err(BenchmarkSuiteIdentityError::DigestMismatch);
        }
        Ok(())
    }
}

fn validate_benchmark_id(benchmark_id: &str) -> Result<(), BenchmarkSuiteIdentityError> {
    if benchmark_id.is_empty() {
        return Err(BenchmarkSuiteIdentityError::EmptyBenchmarkId);
    }
    if benchmark_id.trim() != benchmark_id
        || benchmark_id
            .chars()
            .any(|ch| ch == '\0' || ch == '\n' || ch == '\r' || ch.is_control())
    {
        return Err(BenchmarkSuiteIdentityError::NonCanonicalBenchmarkId {
            benchmark_id: benchmark_id.to_string(),
        });
    }
    Ok(())
}

fn validate_component(
    field: &'static str,
    value: &str,
) -> Result<(), BenchmarkSuiteIdentityError> {
    if value.is_empty()
        || value.trim() != value
        || value
            .chars()
            .any(|ch| ch == '\0' || ch == '\n' || ch == '\r' || ch.is_control())
    {
        return Err(BenchmarkSuiteIdentityError::InvalidComponent {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn compute_digest(
    purpose: &BenchmarkSuitePurpose,
    benchmark_ids: &[String],
) -> Result<String, BenchmarkSuiteIdentityError> {
    let mut hasher = blake3::Hasher::new();
    update_len_prefixed(&mut hasher, BENCHMARK_SUITE_IDENTITY_SCHEMA_VERSION.as_bytes());
    let purpose = purpose.canonical_label()?;
    update_len_prefixed(&mut hasher, purpose.as_bytes());
    hasher.update(&(benchmark_ids.len() as u64).to_le_bytes());
    for benchmark_id in benchmark_ids {
        update_len_prefixed(&mut hasher, benchmark_id.as_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn update_len_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{BenchmarkConfig, BenchmarkResult};

    struct NamedBenchmark(&'static str);

    impl PsychBenchmark for NamedBenchmark {
        fn name(&self) -> &str {
            self.0
        }

        fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
            BenchmarkResult::new(self.0, None)
        }
    }

    #[test]
    fn executable_membership_is_order_independent() {
        let a = NamedBenchmark("Executive::Stroop");
        let b = NamedBenchmark("Memory::NBack");
        let first = ExecutableBenchmarkSuiteIdentity::from_benchmarks(
            BenchmarkSuitePurpose::PaperBattery,
            [&a as &dyn PsychBenchmark, &b as &dyn PsychBenchmark],
        )
        .unwrap();
        let second = ExecutableBenchmarkSuiteIdentity::from_benchmarks(
            BenchmarkSuitePurpose::PaperBattery,
            [&b as &dyn PsychBenchmark, &a as &dyn PsychBenchmark],
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.benchmark_count, 2);
        assert_eq!(
            first.benchmark_ids,
            vec!["Executive::Stroop", "Memory::NBack"]
        );
        first.validate().unwrap();
    }

    #[test]
    fn membership_drift_changes_digest() {
        let baseline = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::FullBattery,
            ["A".to_string(), "B".to_string()],
        )
        .unwrap();
        let changed = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::FullBattery,
            ["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();
        assert_ne!(baseline.digest, changed.digest);
    }

    #[test]
    fn purpose_is_part_of_identity() {
        let full = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::FullBattery,
            ["A".to_string()],
        )
        .unwrap();
        let paper = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::PaperBattery,
            ["A".to_string()],
        )
        .unwrap();
        assert_ne!(full.digest, paper.digest);
    }

    #[test]
    fn duplicate_runtime_ids_fail_closed() {
        let first = NamedBenchmark("Executive::Stroop");
        let second = NamedBenchmark("Executive::Stroop");
        assert_eq!(
            ExecutableBenchmarkSuiteIdentity::from_benchmarks(
                BenchmarkSuitePurpose::RegressionBattery,
                [
                    &first as &dyn PsychBenchmark,
                    &second as &dyn PsychBenchmark,
                ],
            ),
            Err(BenchmarkSuiteIdentityError::DuplicateBenchmarkId {
                benchmark_id: "Executive::Stroop".to_string(),
            })
        );
    }

    #[test]
    fn empty_and_noncanonical_ids_fail_closed() {
        assert_eq!(
            ExecutableBenchmarkSuiteIdentity::from_ids(
                BenchmarkSuitePurpose::FullBattery,
                Vec::<String>::new(),
            ),
            Err(BenchmarkSuiteIdentityError::EmptySuite)
        );
        assert!(matches!(
            ExecutableBenchmarkSuiteIdentity::from_ids(
                BenchmarkSuitePurpose::FullBattery,
                [" Executive::Stroop".to_string()],
            ),
            Err(BenchmarkSuiteIdentityError::NonCanonicalBenchmarkId { .. })
        ));
        assert!(matches!(
            ExecutableBenchmarkSuiteIdentity::from_ids(
                BenchmarkSuitePurpose::FullBattery,
                ["Executive::Stroop\n".to_string()],
            ),
            Err(BenchmarkSuiteIdentityError::NonCanonicalBenchmarkId { .. })
        ));
    }

    #[test]
    fn custom_purpose_must_be_canonical() {
        assert!(ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::Custom("paper-v2".to_string()),
            ["A".to_string()],
        )
        .is_ok());
        assert!(matches!(
            ExecutableBenchmarkSuiteIdentity::from_ids(
                BenchmarkSuitePurpose::Custom(" paper-v2".to_string()),
                ["A".to_string()],
            ),
            Err(BenchmarkSuiteIdentityError::InvalidComponent {
                field: "suite purpose",
                ..
            })
        ));
    }

    #[test]
    fn serialized_receipt_revalidates_and_tampering_fails() {
        let identity = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::MultiSeedRobustness,
            ["A".to_string(), "B".to_string()],
        )
        .unwrap();
        let json = serde_json::to_string(&identity).unwrap();
        let replay: ExecutableBenchmarkSuiteIdentity = serde_json::from_str(&json).unwrap();
        replay.validate().unwrap();

        let mut tampered = replay.clone();
        tampered.benchmark_ids[1] = "C".to_string();
        assert_eq!(
            tampered.validate(),
            Err(BenchmarkSuiteIdentityError::DigestMismatch)
        );
    }

    #[test]
    fn reordered_serialized_membership_is_rejected() {
        let mut identity = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::FullBattery,
            ["A".to_string(), "B".to_string()],
        )
        .unwrap();
        identity.benchmark_ids.swap(0, 1);
        assert_eq!(
            identity.validate(),
            Err(BenchmarkSuiteIdentityError::NonCanonicalOrdering)
        );
    }

    #[test]
    fn count_tampering_is_rejected() {
        let mut identity = ExecutableBenchmarkSuiteIdentity::from_ids(
            BenchmarkSuitePurpose::SatSubset,
            ["A".to_string()],
        )
        .unwrap();
        identity.benchmark_count = 2;
        assert_eq!(
            identity.validate(),
            Err(BenchmarkSuiteIdentityError::BenchmarkCountMismatch {
                declared: 2,
                actual: 1,
            })
        );
    }
}
