// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Binds a regression snapshot to the exact observed executable benchmark population.
//!
//! [`RegressionSnapshot`](crate::harness::snapshot::RegressionSnapshot) predates
//! executable-suite identity. Its `from_report` constructor stores benchmark results
//! in a map, so duplicate benchmark IDs would otherwise collapse silently. This
//! additive wrapper validates the original [`BenchmarkReport`] before that collapse,
//! requires exact population equality with the qualified executable-suite identity,
//! and binds the intended code subject separately.
//!
//! This is a population/consistency theorem only. It does not grant metric-direction,
//! policy, execution, calibration, scientific-validity, or result-quality authority.

use crate::harness::report::BenchmarkReport;
use crate::harness::snapshot::RegressionSnapshot;
use crate::suite_identity::{BenchmarkSuiteIdentityError, ExecutableBenchmarkSuiteIdentity};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const SUITE_BOUND_REGRESSION_SNAPSHOT_SCHEMA_VERSION: &str =
    "psych-regression-snapshot-suite-bound-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteBoundRegressionSnapshot {
    pub schema_version: String,
    /// Exact #3303 executable-population digest.
    pub executable_suite_digest: String,
    /// Intended Git/code subject. This is identity, not proof of process execution.
    pub code_subject: String,
    /// Number of source `BenchmarkReport` result records before map construction.
    pub source_result_count: usize,
    pub snapshot: RegressionSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuiteBoundSnapshotError {
    Suite(BenchmarkSuiteIdentityError),
    UnsupportedSchema,
    NonCanonicalCodeSubject,
    DuplicateReportBenchmark(String),
    MissingBenchmark(String),
    UnexpectedBenchmark(String),
    SuiteDigestMismatch,
    SourceResultCountMismatch {
        declared: usize,
        expected: usize,
    },
    SnapshotPopulationMismatch,
    InvalidSnapshot(Vec<String>),
}

impl From<BenchmarkSuiteIdentityError> for SuiteBoundSnapshotError {
    fn from(value: BenchmarkSuiteIdentityError) -> Self {
        Self::Suite(value)
    }
}

impl fmt::Display for SuiteBoundSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Suite(error) => write!(f, "invalid executable suite identity: {error:?}"),
            Self::UnsupportedSchema => write!(f, "unsupported suite-bound snapshot schema"),
            Self::NonCanonicalCodeSubject => write!(f, "code subject must be non-empty canonical text"),
            Self::DuplicateReportBenchmark(id) => {
                write!(f, "duplicate benchmark result would collapse in snapshot map: {id}")
            }
            Self::MissingBenchmark(id) => write!(f, "suite benchmark missing from report: {id}"),
            Self::UnexpectedBenchmark(id) => write!(f, "report contains benchmark outside suite: {id}"),
            Self::SuiteDigestMismatch => write!(f, "snapshot suite digest does not match executable suite"),
            Self::SourceResultCountMismatch { declared, expected } => write!(
                f,
                "source result count mismatch: declared {declared}, expected {expected}"
            ),
            Self::SnapshotPopulationMismatch => {
                write!(f, "serialized snapshot benchmark population does not match executable suite")
            }
            Self::InvalidSnapshot(errors) => {
                write!(f, "invalid regression snapshot: {}", errors.join("; "))
            }
        }
    }
}

impl std::error::Error for SuiteBoundSnapshotError {}

impl SuiteBoundRegressionSnapshot {
    /// Construct from the uncollapsed report and an already-qualified observed
    /// executable suite identity.
    pub fn from_report(
        report: &BenchmarkReport,
        name: &str,
        suite: &ExecutableBenchmarkSuiteIdentity,
        code_subject: impl Into<String>,
    ) -> Result<Self, SuiteBoundSnapshotError> {
        suite.validate()?;
        let code_subject = code_subject.into();
        validate_code_subject(&code_subject)?;

        let mut observed = BTreeSet::new();
        for result in &report.results {
            if !observed.insert(result.benchmark.as_str()) {
                return Err(SuiteBoundSnapshotError::DuplicateReportBenchmark(
                    result.benchmark.clone(),
                ));
            }
        }

        let expected: BTreeSet<&str> = suite.benchmark_ids.iter().map(String::as_str).collect();
        if let Some(missing) = expected.difference(&observed).next() {
            return Err(SuiteBoundSnapshotError::MissingBenchmark((*missing).to_string()));
        }
        if let Some(extra) = observed.difference(&expected).next() {
            return Err(SuiteBoundSnapshotError::UnexpectedBenchmark((*extra).to_string()));
        }

        let snapshot = RegressionSnapshot::from_report(report, name);
        snapshot
            .validate()
            .map_err(SuiteBoundSnapshotError::InvalidSnapshot)?;

        let bound = Self {
            schema_version: SUITE_BOUND_REGRESSION_SNAPSHOT_SCHEMA_VERSION.to_string(),
            executable_suite_digest: suite.digest.clone(),
            code_subject,
            source_result_count: report.results.len(),
            snapshot,
        };
        bound.validate_against_suite(suite)?;
        Ok(bound)
    }

    /// Revalidate serialized consistency against the observed executable suite.
    ///
    /// This can prove the record is self-consistent. It cannot prove a serialized
    /// record was actually constructed through [`Self::from_report`].
    pub fn validate_against_suite(
        &self,
        suite: &ExecutableBenchmarkSuiteIdentity,
    ) -> Result<(), SuiteBoundSnapshotError> {
        if self.schema_version != SUITE_BOUND_REGRESSION_SNAPSHOT_SCHEMA_VERSION {
            return Err(SuiteBoundSnapshotError::UnsupportedSchema);
        }
        suite.validate()?;
        validate_code_subject(&self.code_subject)?;
        if self.executable_suite_digest != suite.digest {
            return Err(SuiteBoundSnapshotError::SuiteDigestMismatch);
        }
        if self.source_result_count != suite.benchmark_count {
            return Err(SuiteBoundSnapshotError::SourceResultCountMismatch {
                declared: self.source_result_count,
                expected: suite.benchmark_count,
            });
        }
        self.snapshot
            .validate()
            .map_err(SuiteBoundSnapshotError::InvalidSnapshot)?;

        let snapshot_ids: Vec<&str> = self.snapshot.metrics.keys().map(String::as_str).collect();
        let suite_ids: Vec<&str> = suite.benchmark_ids.iter().map(String::as_str).collect();
        if snapshot_ids != suite_ids {
            return Err(SuiteBoundSnapshotError::SnapshotPopulationMismatch);
        }
        Ok(())
    }
}

fn validate_code_subject(code_subject: &str) -> Result<(), SuiteBoundSnapshotError> {
    if code_subject.is_empty()
        || code_subject.trim() != code_subject
        || code_subject.chars().any(char::is_control)
    {
        return Err(SuiteBoundSnapshotError::NonCanonicalCodeSubject);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{BenchmarkConfig, BenchmarkResult, PsychBenchmark};
    use crate::suite_identity::BenchmarkSuitePurpose;

    struct NamedBenchmark(&'static str);

    impl PsychBenchmark for NamedBenchmark {
        fn name(&self) -> &str {
            self.0
        }

        fn run(&self, _config: &BenchmarkConfig) -> BenchmarkResult {
            BenchmarkResult::new(self.0, None)
        }
    }

    fn report_and_suite() -> (BenchmarkReport, ExecutableBenchmarkSuiteIdentity) {
        let a = NamedBenchmark("A");
        let b = NamedBenchmark("B");
        let config = BenchmarkConfig::default();
        let mut report = BenchmarkReport::new();
        report.add(a.run(&config));
        report.add(b.run(&config));
        let suite = ExecutableBenchmarkSuiteIdentity::from_benchmarks(
            BenchmarkSuitePurpose::RegressionBattery,
            [&a as &dyn PsychBenchmark, &b as &dyn PsychBenchmark],
        )
        .unwrap();
        (report, suite)
    }

    #[test]
    fn suite_bound_snapshot_requires_exact_report_population() {
        let (report, suite) = report_and_suite();
        let bound = SuiteBoundRegressionSnapshot::from_report(
            &report,
            "baseline",
            &suite,
            "git:0123456789abcdef",
        )
        .unwrap();
        assert_eq!(bound.source_result_count, 2);
        assert_eq!(bound.executable_suite_digest, suite.digest);
        bound.validate_against_suite(&suite).unwrap();
    }

    #[test]
    fn duplicate_report_results_fail_before_map_collapse() {
        let a = NamedBenchmark("A");
        let config = BenchmarkConfig::default();
        let mut report = BenchmarkReport::new();
        report.add(a.run(&config));
        report.add(a.run(&config));
        let suite = ExecutableBenchmarkSuiteIdentity::from_benchmarks(
            BenchmarkSuitePurpose::RegressionBattery,
            [&a as &dyn PsychBenchmark],
        )
        .unwrap();
        assert!(matches!(
            SuiteBoundRegressionSnapshot::from_report(
                &report,
                "duplicate",
                &suite,
                "git:0123456789abcdef",
            ),
            Err(SuiteBoundSnapshotError::DuplicateReportBenchmark(id)) if id == "A"
        ));
    }

    #[test]
    fn missing_and_unexpected_report_members_fail_closed() {
        let a = NamedBenchmark("A");
        let b = NamedBenchmark("B");
        let c = NamedBenchmark("C");
        let config = BenchmarkConfig::default();
        let suite = ExecutableBenchmarkSuiteIdentity::from_benchmarks(
            BenchmarkSuitePurpose::RegressionBattery,
            [&a as &dyn PsychBenchmark, &b as &dyn PsychBenchmark],
        )
        .unwrap();

        let mut missing = BenchmarkReport::new();
        missing.add(a.run(&config));
        assert!(matches!(
            SuiteBoundRegressionSnapshot::from_report(
                &missing,
                "missing",
                &suite,
                "git:0123456789abcdef",
            ),
            Err(SuiteBoundSnapshotError::MissingBenchmark(id)) if id == "B"
        ));

        let suite_a = ExecutableBenchmarkSuiteIdentity::from_benchmarks(
            BenchmarkSuitePurpose::RegressionBattery,
            [&a as &dyn PsychBenchmark],
        )
        .unwrap();
        let mut extra = BenchmarkReport::new();
        extra.add(a.run(&config));
        extra.add(c.run(&config));
        assert!(matches!(
            SuiteBoundRegressionSnapshot::from_report(
                &extra,
                "extra",
                &suite_a,
                "git:0123456789abcdef",
            ),
            Err(SuiteBoundSnapshotError::UnexpectedBenchmark(id)) if id == "C"
        ));
    }

    #[test]
    fn serialized_population_suite_digest_count_and_code_subject_revalidate() {
        let (report, suite) = report_and_suite();
        let original = SuiteBoundRegressionSnapshot::from_report(
            &report,
            "baseline",
            &suite,
            "git:0123456789abcdef",
        )
        .unwrap();
        let json = serde_json::to_vec(&original).unwrap();
        let decoded: SuiteBoundRegressionSnapshot = serde_json::from_slice(&json).unwrap();
        decoded.validate_against_suite(&suite).unwrap();

        let mut wrong_digest = decoded.clone();
        wrong_digest.executable_suite_digest = "0".repeat(64);
        assert_eq!(
            wrong_digest.validate_against_suite(&suite),
            Err(SuiteBoundSnapshotError::SuiteDigestMismatch)
        );

        let mut wrong_count = decoded.clone();
        wrong_count.source_result_count += 1;
        assert!(matches!(
            wrong_count.validate_against_suite(&suite),
            Err(SuiteBoundSnapshotError::SourceResultCountMismatch { .. })
        ));

        let mut wrong_population = decoded.clone();
        wrong_population.snapshot.metrics.remove("B");
        assert_eq!(
            wrong_population.validate_against_suite(&suite),
            Err(SuiteBoundSnapshotError::SnapshotPopulationMismatch)
        );

        let mut wrong_subject = decoded;
        wrong_subject.code_subject = " git:bad".into();
        assert_eq!(
            wrong_subject.validate_against_suite(&suite),
            Err(SuiteBoundSnapshotError::NonCanonicalCodeSubject)
        );
    }
}
