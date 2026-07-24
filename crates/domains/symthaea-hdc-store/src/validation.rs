// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Empirical ANN validation against the store's exact-search contract.

use std::collections::HashSet;
use std::fmt::Write as _;

use symthaea_core::hdc::BinaryHV;

use crate::{ApproximateSearchOptions, HdcStore};

/// Per-query comparison between exact and approximate top-k results.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnQueryResult {
    pub query_index: usize,
    pub recall: f64,
    pub examined: usize,
    pub total_live: usize,
    pub candidate_fraction: f64,
    pub exhaustive: bool,
    pub fell_back_to_exact: bool,
    pub exact_ids: Vec<u64>,
    pub approximate_ids: Vec<u64>,
}

/// Aggregate measurements for one top-k value.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnValidationReport {
    pub top_k: usize,
    pub query_count: usize,
    pub mean_recall: f64,
    pub worst_recall: f64,
    pub exhaustive_query_fraction: f64,
    pub fallback_query_fraction: f64,
    pub mean_candidate_fraction: f64,
    pub p50_candidate_fraction: f64,
    pub p95_candidate_fraction: f64,
    pub total_examined: u64,
    pub queries: Vec<AnnQueryResult>,
}

impl AnnValidationReport {
    /// Evaluate one top-k setting against exact ground truth.
    pub fn evaluate(
        store: &HdcStore,
        queries: &[BinaryHV],
        top_k: usize,
        options: ApproximateSearchOptions,
    ) -> Self {
        let mut results = Vec::with_capacity(queries.len());
        for (query_index, query) in queries.iter().enumerate() {
            let exact = store.scan_similar(query, top_k);
            let approximate = store.scan_similar_approx(query, top_k, options);
            let exact_ids: Vec<u64> = exact.into_iter().map(|(id, _)| id).collect();
            let approximate_ids: Vec<u64> = approximate
                .neighbors
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let recall = recall_at_k(&exact_ids, &approximate_ids);
            let candidate_fraction = fraction(approximate.examined, approximate.total_live);
            results.push(AnnQueryResult {
                query_index,
                recall,
                examined: approximate.examined,
                total_live: approximate.total_live,
                candidate_fraction,
                exhaustive: approximate.exact,
                fell_back_to_exact: approximate.fell_back_to_exact,
                exact_ids,
                approximate_ids,
            });
        }
        Self::from_query_results(top_k, results)
    }

    fn from_query_results(top_k: usize, queries: Vec<AnnQueryResult>) -> Self {
        if queries.is_empty() {
            return Self {
                top_k,
                query_count: 0,
                mean_recall: 0.0,
                worst_recall: 0.0,
                exhaustive_query_fraction: 0.0,
                fallback_query_fraction: 0.0,
                mean_candidate_fraction: 0.0,
                p50_candidate_fraction: 0.0,
                p95_candidate_fraction: 0.0,
                total_examined: 0,
                queries,
            };
        }

        let query_count = queries.len();
        let mean_recall =
            queries.iter().map(|query| query.recall).sum::<f64>() / query_count as f64;
        let worst_recall = queries
            .iter()
            .map(|query| query.recall)
            .fold(1.0f64, f64::min);
        let exhaustive_query_fraction =
            queries.iter().filter(|query| query.exhaustive).count() as f64 / query_count as f64;
        let fallback_query_fraction = queries
            .iter()
            .filter(|query| query.fell_back_to_exact)
            .count() as f64
            / query_count as f64;
        let mean_candidate_fraction = queries
            .iter()
            .map(|query| query.candidate_fraction)
            .sum::<f64>()
            / query_count as f64;
        let mut candidate_fractions: Vec<f64> = queries
            .iter()
            .map(|query| query.candidate_fraction)
            .collect();
        candidate_fractions.sort_by(f64::total_cmp);
        let total_examined = queries.iter().map(|query| query.examined as u64).sum();

        Self {
            top_k,
            query_count,
            mean_recall,
            worst_recall,
            exhaustive_query_fraction,
            fallback_query_fraction,
            mean_candidate_fraction,
            p50_candidate_fraction: percentile(&candidate_fractions, 0.50),
            p95_candidate_fraction: percentile(&candidate_fractions, 0.95),
            total_examined,
            queries,
        }
    }

    /// Evaluate the report against explicit CI/release thresholds.
    pub fn gate(&self, thresholds: AnnValidationThresholds) -> AnnGateResult {
        let mut failures = Vec::new();
        if self.query_count < thresholds.minimum_queries {
            failures.push(format!(
                "query_count {} is below required minimum {}",
                self.query_count, thresholds.minimum_queries
            ));
        }
        if self.mean_recall < thresholds.minimum_mean_recall {
            failures.push(format!(
                "mean recall {:.6} is below required {:.6}",
                self.mean_recall, thresholds.minimum_mean_recall
            ));
        }
        if self.worst_recall < thresholds.minimum_worst_recall {
            failures.push(format!(
                "worst recall {:.6} is below required {:.6}",
                self.worst_recall, thresholds.minimum_worst_recall
            ));
        }
        if self.mean_candidate_fraction > thresholds.maximum_mean_candidate_fraction {
            failures.push(format!(
                "mean candidate fraction {:.6} exceeds maximum {:.6}",
                self.mean_candidate_fraction, thresholds.maximum_mean_candidate_fraction
            ));
        }
        if self.exhaustive_query_fraction > thresholds.maximum_exhaustive_query_fraction {
            failures.push(format!(
                "exhaustive query fraction {:.6} exceeds maximum {:.6}",
                self.exhaustive_query_fraction, thresholds.maximum_exhaustive_query_fraction
            ));
        }
        if self.fallback_query_fraction > thresholds.maximum_fallback_query_fraction {
            failures.push(format!(
                "fallback query fraction {:.6} exceeds maximum {:.6}",
                self.fallback_query_fraction, thresholds.maximum_fallback_query_fraction
            ));
        }
        AnnGateResult {
            passed: failures.is_empty(),
            failures,
        }
    }

    /// Render deterministic CSV suitable for CI evidence artifacts.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "query_index,top_k,recall,examined,total_live,candidate_fraction,exhaustive,fell_back_to_exact,exact_ids,approximate_ids\n",
        );
        for query in &self.queries {
            let exact_ids = join_ids(&query.exact_ids);
            let approximate_ids = join_ids(&query.approximate_ids);
            writeln!(
                csv,
                "{},{},{:.9},{},{},{:.9},{},{},\"{}\",\"{}\"",
                query.query_index,
                self.top_k,
                query.recall,
                query.examined,
                query.total_live,
                query.candidate_fraction,
                query.exhaustive,
                query.fell_back_to_exact,
                exact_ids,
                approximate_ids
            )
            .expect("writing to String cannot fail");
        }
        csv
    }
}

/// A set of reports for multiple top-k values over the same queries.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnValidationSuite {
    pub reports: Vec<AnnValidationReport>,
}

impl AnnValidationSuite {
    pub fn evaluate(
        store: &HdcStore,
        queries: &[BinaryHV],
        top_ks: &[usize],
        options: ApproximateSearchOptions,
    ) -> Self {
        let mut unique_top_ks = top_ks.to_vec();
        unique_top_ks.sort_unstable();
        unique_top_ks.dedup();
        let reports = unique_top_ks
            .into_iter()
            .map(|top_k| AnnValidationReport::evaluate(store, queries, top_k, options))
            .collect();
        Self { reports }
    }

    /// All reports must pass the same thresholds.
    pub fn gate(&self, thresholds: AnnValidationThresholds) -> AnnGateResult {
        let mut failures = Vec::new();
        if self.reports.is_empty() {
            failures.push("validation suite contains no top-k reports".into());
        }
        for report in &self.reports {
            let result = report.gate(thresholds);
            failures.extend(
                result
                    .failures
                    .into_iter()
                    .map(|failure| format!("k={}: {failure}", report.top_k)),
            );
        }
        AnnGateResult {
            passed: failures.is_empty(),
            failures,
        }
    }
}

/// Explicit acceptance thresholds for empirical ANN reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnValidationThresholds {
    pub minimum_queries: usize,
    pub minimum_mean_recall: f64,
    pub minimum_worst_recall: f64,
    pub maximum_mean_candidate_fraction: f64,
    pub maximum_exhaustive_query_fraction: f64,
    pub maximum_fallback_query_fraction: f64,
}

impl Default for AnnValidationThresholds {
    fn default() -> Self {
        Self {
            minimum_queries: 100,
            minimum_mean_recall: 0.95,
            minimum_worst_recall: 0.80,
            maximum_mean_candidate_fraction: 0.50,
            maximum_exhaustive_query_fraction: 0.50,
            maximum_fallback_query_fraction: 0.05,
        }
    }
}

/// Result of applying validation thresholds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnGateResult {
    pub passed: bool,
    pub failures: Vec<String>,
}

fn recall_at_k(exact_ids: &[u64], approximate_ids: &[u64]) -> f64 {
    if exact_ids.is_empty() {
        return 1.0;
    }
    let approximate: HashSet<u64> = approximate_ids.iter().copied().collect();
    let recovered = exact_ids
        .iter()
        .filter(|id| approximate.contains(*id))
        .count();
    recovered as f64 / exact_ids.len() as f64
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (quantile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn join_ids(ids: &[u64]) -> String {
    ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StoreConfig;
    use tempfile::tempdir;

    fn populated_store(count: u64) -> (tempfile::TempDir, HdcStore) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("validation.hdc");
        let mut store = HdcStore::create(path, StoreConfig::default()).unwrap();
        for id in 0..count {
            store.append(id, &BinaryHV::random(id)).unwrap();
        }
        (dir, store)
    }

    #[test]
    fn exact_fallback_produces_perfect_recall() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("validation.hdc");
        let mut store = HdcStore::create(
            path,
            StoreConfig {
                lsh_bands: 1,
                lsh_rows: 32,
                ..StoreConfig::default()
            },
        )
        .unwrap();
        for id in 0..32 {
            store.append(id, &BinaryHV::random(id)).unwrap();
        }
        let queries: Vec<_> = (0..8).map(BinaryHV::random).collect();
        let report = AnnValidationReport::evaluate(
            &store,
            &queries,
            5,
            ApproximateSearchOptions {
                candidate_multiplier: usize::MAX,
                fallback_on_empty: true,
            },
        );
        assert_eq!(report.query_count, 8);
        assert_eq!(report.mean_recall, 1.0);
        assert_eq!(report.worst_recall, 1.0);
        assert_eq!(report.exhaustive_query_fraction, 1.0);
        assert_eq!(report.fallback_query_fraction, 1.0);
        assert_eq!(report.mean_candidate_fraction, 1.0);
    }

    #[test]
    fn self_queries_recover_top_one_without_fallback() {
        let (_dir, store) = populated_store(32);
        let queries: Vec<_> = (0..16).map(BinaryHV::random).collect();
        let report = AnnValidationReport::evaluate(
            &store,
            &queries,
            1,
            ApproximateSearchOptions {
                candidate_multiplier: 0,
                fallback_on_empty: false,
            },
        );
        assert_eq!(report.mean_recall, 1.0);
        assert_eq!(report.worst_recall, 1.0);
        assert!(report.mean_candidate_fraction <= 1.0);
        assert!(report.p50_candidate_fraction <= report.p95_candidate_fraction);
    }

    #[test]
    fn suite_sorts_and_deduplicates_top_k_values() {
        let (_dir, store) = populated_store(8);
        let queries = vec![BinaryHV::random(1)];
        let suite = AnnValidationSuite::evaluate(
            &store,
            &queries,
            &[10, 1, 5, 1],
            ApproximateSearchOptions::default(),
        );
        assert_eq!(
            suite
                .reports
                .iter()
                .map(|report| report.top_k)
                .collect::<Vec<_>>(),
            vec![1, 5, 10]
        );
    }

    #[test]
    fn gate_reports_each_failed_dimension() {
        let report = AnnValidationReport::from_query_results(
            5,
            vec![AnnQueryResult {
                query_index: 0,
                recall: 0.5,
                examined: 90,
                total_live: 100,
                candidate_fraction: 0.9,
                exhaustive: true,
                fell_back_to_exact: true,
                exact_ids: vec![1, 2],
                approximate_ids: vec![1, 3],
            }],
        );
        let gate = report.gate(AnnValidationThresholds {
            minimum_queries: 2,
            minimum_mean_recall: 0.9,
            minimum_worst_recall: 0.8,
            maximum_mean_candidate_fraction: 0.5,
            maximum_exhaustive_query_fraction: 0.1,
            maximum_fallback_query_fraction: 0.1,
        });
        assert!(!gate.passed);
        assert_eq!(gate.failures.len(), 6);
    }

    #[test]
    fn csv_is_deterministic_and_contains_ids() {
        let report = AnnValidationReport::from_query_results(
            2,
            vec![AnnQueryResult {
                query_index: 0,
                recall: 0.5,
                examined: 3,
                total_live: 10,
                candidate_fraction: 0.3,
                exhaustive: false,
                fell_back_to_exact: false,
                exact_ids: vec![1, 2],
                approximate_ids: vec![2, 9],
            }],
        );
        let csv = report.to_csv();
        assert!(csv.contains("0,2,0.500000000,3,10,0.300000000,false,false"));
        assert!(csv.contains("\"1;2\",\"2;9\""));
    }
    #[test]
    fn empty_suite_fails_closed() {
        let suite = AnnValidationSuite {
            reports: Vec::new(),
        };
        let gate = suite.gate(AnnValidationThresholds::default());
        assert!(!gate.passed);
        assert_eq!(gate.failures.len(), 1);
    }
}
