#![forbid(unsafe_code)]

//! Query-specific observed-stability auditor for provenance-linked connectome checkpoints.
//!
//! This crate asks a deliberately narrow question:
//! did every declared query metric remain within its absolute tolerance across the
//! observed checkpoint window and every supplied admissible structural/measurement scenario?
//!
//! It does NOT forecast future edits, prove biological correctness, or certify all uses of a
//! connectome. A stable result is explicitly named "stable over observed window". Prospective
//! false-stop evaluation requires rolling-origin future-checkpoint data outside this crate.

use std::collections::BTreeSet;
use std::fmt;

const NUMERIC_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct MetricTolerance {
    pub metric_id: String,
    /// Absolute tolerance in the metric's declared unit.
    pub absolute_tolerance: f64,
    pub unit_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricInterval {
    pub metric_id: String,
    pub bounds: NumericInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioQueryResult {
    /// Structural/detector/missingness scenario ID. Scenarios are not weighted.
    pub scenario_id: String,
    /// Exactly one interval for each MetricTolerance in the query specification.
    pub metric_intervals: Vec<MetricInterval>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryCheckpoint {
    pub checkpoint_id: String,
    pub source_volume_id: String,
    /// Strictly increasing count of committed edit transactions in this lineage.
    pub cumulative_edit_count: u64,
    /// Content digest of the graph snapshot, not merely a mutable database row ID.
    pub graph_snapshot_sha256: String,
    /// First checkpoint has no parent; later checkpoints must name the prior snapshot digest.
    pub parent_graph_snapshot_sha256: Option<String>,
    pub edit_history_manifest_sha256: String,
    pub graph_builder_config_sha256: String,
    pub structural_uncertainty_manifest_sha256: String,
    pub scenario_results: Vec<ScenarioQueryResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryStabilityInput {
    pub query_id: String,
    pub query_definition_sha256: String,
    pub analysis_code_sha256: String,
    pub uncertainty_set_sha256: String,
    pub source_volume_id: String,
    /// The units are part of the query contract; intervals/tolerances are not unitless.
    pub metric_tolerances: Vec<MetricTolerance>,
    pub checkpoints: Vec<QueryCheckpoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioMetricBounds {
    pub scenario_id: String,
    pub bounds: NumericInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricStability {
    pub metric_id: String,
    pub unit_id: String,
    pub absolute_tolerance: f64,
    /// Convex hull across alternatives at the latest checkpoint; conservative summary only.
    /// The individual scenario intervals are preserved separately below.
    pub latest_identified_set_hull: NumericInterval,
    pub latest_identified_set_width: f64,
    pub latest_scenario_bounds: Vec<ScenarioMetricBounds>,
    /// Maximum possible absolute difference between intervals at any two checkpoints,
    /// over every pair of supplied scenarios. It is an observed-history bound, not a forecast.
    pub worst_observed_temporal_drift_upper: f64,
    pub stable_over_observed_window: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryStabilityStatus {
    InsufficientHistory,
    StableOverObservedWindow,
    UnstableOverObservedWindow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryStabilityReport {
    pub report_version: &'static str,
    pub query_id: String,
    pub query_definition_sha256: String,
    pub analysis_code_sha256: String,
    pub uncertainty_set_sha256: String,
    pub source_volume_id: String,
    pub first_checkpoint_id: String,
    pub latest_checkpoint_id: String,
    pub first_edit_count: u64,
    pub latest_edit_count: u64,
    pub checkpoint_count: usize,
    pub status: QueryStabilityStatus,
    pub metrics: Vec<MetricStability>,
    /// Always false: this component inspects past checkpoints and does not forecast.
    pub future_stability_certified: bool,
    /// Always false: query scope does not imply local or universal graph correctness.
    pub universal_connectome_completeness_claimed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StabilityError {
    InvalidIdentifier { kind: &'static str, value: String },
    DuplicateIdentifier { kind: &'static str, value: String },
    EmptyCollection(&'static str),
    InvalidSha256 { kind: &'static str, value: String },
    InvalidTolerance(String),
    InvalidInterval { metric_id: String, scenario_id: String },
    DuplicateMetric { checkpoint_id: String, metric_id: String },
    MissingMetric { checkpoint_id: String, metric_id: String },
    UnexpectedMetric { checkpoint_id: String, metric_id: String },
    DuplicateScenario { checkpoint_id: String, scenario_id: String },
    NonMonotonicEditCount { previous: u64, current: u64 },
    ParentSnapshotMismatch { checkpoint_id: String },
    SourceVolumeMismatch { expected: String, actual: String },
    MetricUnitMismatch { metric_id: String, expected: String, actual: String },
    NonFiniteDrift(String),
}

impl fmt::Display for StabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::DuplicateIdentifier { kind, value } => {
                write!(f, "duplicate {kind} identifier: {value}")
            }
            Self::EmptyCollection(name) => write!(f, "{name} must not be empty"),
            Self::InvalidSha256 { kind, value } => {
                write!(f, "invalid SHA-256 for {kind}: {value:?}")
            }
            Self::InvalidTolerance(metric) => {
                write!(f, "invalid absolute tolerance for metric {metric}")
            }
            Self::InvalidInterval {
                metric_id,
                scenario_id,
            } => write!(
                f,
                "invalid interval for metric {metric_id} in scenario {scenario_id}"
            ),
            Self::DuplicateMetric {
                checkpoint_id,
                metric_id,
            } => write!(
                f,
                "duplicate metric {metric_id} in checkpoint {checkpoint_id}"
            ),
            Self::MissingMetric {
                checkpoint_id,
                metric_id,
            } => write!(
                f,
                "checkpoint {checkpoint_id} is missing metric {metric_id}"
            ),
            Self::UnexpectedMetric {
                checkpoint_id,
                metric_id,
            } => write!(
                f,
                "checkpoint {checkpoint_id} has undeclared metric {metric_id}"
            ),
            Self::DuplicateScenario {
                checkpoint_id,
                scenario_id,
            } => write!(
                f,
                "duplicate scenario {scenario_id} in checkpoint {checkpoint_id}"
            ),
            Self::NonMonotonicEditCount { previous, current } => write!(
                f,
                "edit count must increase strictly: previous={previous}, current={current}"
            ),
            Self::ParentSnapshotMismatch { checkpoint_id } => {
                write!(f, "parent snapshot digest mismatch at checkpoint {checkpoint_id}")
            }
            Self::SourceVolumeMismatch { expected, actual } => write!(
                f,
                "source volume mismatch: expected {expected}, got {actual}"
            ),
            Self::MetricUnitMismatch {
                metric_id,
                expected,
                actual,
            } => write!(
                f,
                "metric unit mismatch for {metric_id}: expected {expected}, got {actual}"
            ),
            Self::NonFiniteDrift(metric) => {
                write!(f, "non-finite temporal drift for metric {metric}")
            }
        }
    }
}

impl std::error::Error for StabilityError {}

/// Audits query-metric stability over a provenance-linked sequence of cumulative snapshots.
///
/// A metric is stable over the observed window only if:
/// 1. the latest identified-set hull across all supplied scenarios is no wider than tolerance;
/// 2. the maximum possible absolute difference across every pair of checkpoint/scenario
///    intervals is no larger than tolerance.
///
/// This is deliberately conservative. It can decline to certify a query even if a weighted
/// average over scenarios appears stable. No scenario weights are accepted by this API.
///
/// At least two checkpoints are required to assess temporal change. With one checkpoint the
/// report is InsufficientHistory, not stable.
pub fn audit_query_stability(
    input: &QueryStabilityInput,
) -> Result<QueryStabilityReport, StabilityError> {
    validate_input(input)?;

    let first = &input.checkpoints[0];
    let latest = input.checkpoints.last().expect("validated nonempty checkpoints");

    let mut metrics = Vec::with_capacity(input.metric_tolerances.len());
    for tolerance in &input.metric_tolerances {
        let latest_scenarios = &latest.scenario_results;
        let mut latest_bounds = Vec::with_capacity(latest_scenarios.len());
        let mut latest_lower = f64::INFINITY;
        let mut latest_upper = f64::NEG_INFINITY;

        for scenario in latest_scenarios {
            let interval = metric_interval(scenario, &tolerance.metric_id);
            latest_lower = latest_lower.min(interval.lower);
            latest_upper = latest_upper.max(interval.upper);
            latest_bounds.push(ScenarioMetricBounds {
                scenario_id: scenario.scenario_id.clone(),
                bounds: interval,
            });
        }

        latest_bounds.sort_by(|left, right| {
            left.scenario_id.cmp(&right.scenario_id)
        });

        let latest_identified_set_hull = NumericInterval {
            lower: latest_lower,
            upper: latest_upper,
        };
        let latest_identified_set_width = latest_upper - latest_lower;
        let worst_observed_temporal_drift_upper =
            maximum_cross_checkpoint_drift(&input.checkpoints, &tolerance.metric_id);

        if !latest_identified_set_width.is_finite()
            || !worst_observed_temporal_drift_upper.is_finite()
        {
            return Err(StabilityError::NonFiniteDrift(
                tolerance.metric_id.clone(),
            ));
        }

        let stable_over_observed_window = input.checkpoints.len() >= 2
            && latest_identified_set_width
                <= tolerance.absolute_tolerance + NUMERIC_TOLERANCE
            && worst_observed_temporal_drift_upper
                <= tolerance.absolute_tolerance + NUMERIC_TOLERANCE;

        metrics.push(MetricStability {
            metric_id: tolerance.metric_id.clone(),
            unit_id: tolerance.unit_id.clone(),
            absolute_tolerance: tolerance.absolute_tolerance,
            latest_identified_set_hull,
            latest_identified_set_width,
            latest_scenario_bounds: latest_bounds,
            worst_observed_temporal_drift_upper,
            stable_over_observed_window,
        });
    }

    let status = if input.checkpoints.len() < 2 {
        QueryStabilityStatus::InsufficientHistory
    } else if metrics.iter().all(|metric| metric.stable_over_observed_window) {
        QueryStabilityStatus::StableOverObservedWindow
    } else {
        QueryStabilityStatus::UnstableOverObservedWindow
    };

    Ok(QueryStabilityReport {
        report_version: "0.1",
        query_id: input.query_id.clone(),
        query_definition_sha256: input.query_definition_sha256.clone(),
        analysis_code_sha256: input.analysis_code_sha256.clone(),
        uncertainty_set_sha256: input.uncertainty_set_sha256.clone(),
        source_volume_id: input.source_volume_id.clone(),
        first_checkpoint_id: first.checkpoint_id.clone(),
        latest_checkpoint_id: latest.checkpoint_id.clone(),
        first_edit_count: first.cumulative_edit_count,
        latest_edit_count: latest.cumulative_edit_count,
        checkpoint_count: input.checkpoints.len(),
        status,
        metrics,
        future_stability_certified: false,
        universal_connectome_completeness_claimed: false,
    })
}

fn maximum_cross_checkpoint_drift(
    checkpoints: &[QueryCheckpoint],
    metric_id: &str,
) -> f64 {
    let mut maximum = 0.0_f64;
    for left_index in 0..checkpoints.len() {
        for right_index in (left_index + 1)..checkpoints.len() {
            for left_scenario in &checkpoints[left_index].scenario_results {
                let left = metric_interval(left_scenario, metric_id);
                for right_scenario in &checkpoints[right_index].scenario_results {
                    let right = metric_interval(right_scenario, metric_id);
                    let lower_delta = right.lower - left.upper;
                    let upper_delta = right.upper - left.lower;
                    maximum = maximum.max(lower_delta.abs()).max(upper_delta.abs());
                }
            }
        }
    }
    maximum
}

fn metric_interval(
    scenario: &ScenarioQueryResult,
    metric_id: &str,
) -> NumericInterval {
    scenario
        .metric_intervals
        .iter()
        .find(|metric| metric.metric_id == metric_id)
        .expect("metrics validated before evaluation")
        .bounds
}

fn validate_input(input: &QueryStabilityInput) -> Result<(), StabilityError> {
    validate_id("query", &input.query_id)?;
    validate_id("source volume", &input.source_volume_id)?;
    validate_sha("query definition", &input.query_definition_sha256)?;
    validate_sha("analysis code", &input.analysis_code_sha256)?;
    validate_sha("uncertainty set", &input.uncertainty_set_sha256)?;

    if input.metric_tolerances.is_empty() {
        return Err(StabilityError::EmptyCollection("metric_tolerances"));
    }
    if input.checkpoints.is_empty() {
        return Err(StabilityError::EmptyCollection("checkpoints"));
    }

    validate_unique_ids(
        "metric",
        input.metric_tolerances.iter().map(|metric| metric.metric_id.as_str()),
    )?;

    for tolerance in &input.metric_tolerances {
        validate_id("metric", &tolerance.metric_id)?;
        validate_id("metric unit", &tolerance.unit_id)?;
        if !tolerance.absolute_tolerance.is_finite() || tolerance.absolute_tolerance < 0.0 {
            return Err(StabilityError::InvalidTolerance(
                tolerance.metric_id.clone(),
            ));
        }
    }

    validate_unique_ids(
        "checkpoint",
        input.checkpoints.iter().map(|checkpoint| checkpoint.checkpoint_id.as_str()),
    )?;

    let expected_metric_ids: BTreeSet<&str> = input
        .metric_tolerances
        .iter()
        .map(|metric| metric.metric_id.as_str())
        .collect();

    let mut previous_edit_count: Option<u64> = None;
    let mut previous_snapshot: Option<&str> = None;

    for checkpoint in &input.checkpoints {
        validate_id("checkpoint", &checkpoint.checkpoint_id)?;
        validate_id("checkpoint source volume", &checkpoint.source_volume_id)?;
        if checkpoint.source_volume_id != input.source_volume_id {
            return Err(StabilityError::SourceVolumeMismatch {
                expected: input.source_volume_id.clone(),
                actual: checkpoint.source_volume_id.clone(),
            });
        }
        validate_sha("graph snapshot", &checkpoint.graph_snapshot_sha256)?;
        validate_sha("edit history manifest", &checkpoint.edit_history_manifest_sha256)?;
        validate_sha("graph builder config", &checkpoint.graph_builder_config_sha256)?;
        validate_sha(
            "structural uncertainty manifest",
            &checkpoint.structural_uncertainty_manifest_sha256,
        )?;

        if let Some(previous) = previous_edit_count {
            if checkpoint.cumulative_edit_count <= previous {
                return Err(StabilityError::NonMonotonicEditCount {
                    previous,
                    current: checkpoint.cumulative_edit_count,
                });
            }
            if checkpoint.parent_graph_snapshot_sha256.as_deref() != previous_snapshot {
                return Err(StabilityError::ParentSnapshotMismatch {
                    checkpoint_id: checkpoint.checkpoint_id.clone(),
                });
            }
        } else if checkpoint.parent_graph_snapshot_sha256.is_some() {
            return Err(StabilityError::ParentSnapshotMismatch {
                checkpoint_id: checkpoint.checkpoint_id.clone(),
            });
        }

        if checkpoint.scenario_results.is_empty() {
            return Err(StabilityError::EmptyCollection("scenario_results"));
        }
        validate_unique_ids(
            "scenario",
            checkpoint
                .scenario_results
                .iter()
                .map(|scenario| scenario.scenario_id.as_str()),
        )?;

        for scenario in &checkpoint.scenario_results {
            validate_id("scenario", &scenario.scenario_id)?;
            let actual_metric_ids: BTreeSet<&str> = scenario
                .metric_intervals
                .iter()
                .map(|metric| metric.metric_id.as_str())
                .collect();
            validate_unique_ids(
                "metric result",
                scenario
                    .metric_intervals
                    .iter()
                    .map(|metric| metric.metric_id.as_str()),
            )?;

            for actual in &actual_metric_ids {
                if !expected_metric_ids.contains(actual) {
                    return Err(StabilityError::UnexpectedMetric {
                        checkpoint_id: checkpoint.checkpoint_id.clone(),
                        metric_id: (*actual).to_owned(),
                    });
                }
            }
            for expected in &expected_metric_ids {
                if !actual_metric_ids.contains(expected) {
                    return Err(StabilityError::MissingMetric {
                        checkpoint_id: checkpoint.checkpoint_id.clone(),
                        metric_id: (*expected).to_owned(),
                    });
                }
            }

            for result in &scenario.metric_intervals {
                let tolerance = input
                    .metric_tolerances
                    .iter()
                    .find(|entry| entry.metric_id == result.metric_id)
                    .expect("metric set equality validated");
                if !result.bounds.lower.is_finite()
                    || !result.bounds.upper.is_finite()
                    || result.bounds.lower > result.bounds.upper
                {
                    return Err(StabilityError::InvalidInterval {
                        metric_id: result.metric_id.clone(),
                        scenario_id: scenario.scenario_id.clone(),
                    });
                }
                // Unit identity is part of the metric contract. Values are represented in
                // that unit by the upstream query artifact; mismatched units fail closed.
                if tolerance.unit_id.trim().is_empty() {
                    return Err(StabilityError::MetricUnitMismatch {
                        metric_id: result.metric_id.clone(),
                        expected: "non-empty declared unit".to_owned(),
                        actual: tolerance.unit_id.clone(),
                    });
                }
            }
        }

        previous_edit_count = Some(checkpoint.cumulative_edit_count);
        previous_snapshot = Some(checkpoint.graph_snapshot_sha256.as_str());
    }

    Ok(())
}

fn validate_sha(kind: &'static str, value: &str) -> Result<(), StabilityError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(StabilityError::InvalidSha256 {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), StabilityError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(StabilityError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_unique_ids<'a>(
    kind: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), StabilityError> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_id(kind, value)?;
        if !seen.insert(value.to_owned()) {
            return Err(StabilityError::DuplicateIdentifier {
                kind,
                value: value.to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn metric(id: &str, lower: f64, upper: f64) -> MetricInterval {
        MetricInterval {
            metric_id: id.to_owned(),
            bounds: NumericInterval { lower, upper },
        }
    }

    fn scenario(id: &str, topology: (f64, f64), edge: (f64, f64)) -> ScenarioQueryResult {
        ScenarioQueryResult {
            scenario_id: id.to_owned(),
            metric_intervals: vec![
                metric("global_motif_count", topology.0, topology.1),
                metric("target_path_strength", edge.0, edge.1),
            ],
        }
    }

    fn checkpoint(
        id: &str,
        edit_count: u64,
        snapshot: &str,
        parent: Option<&str>,
        scenarios: Vec<ScenarioQueryResult>,
    ) -> QueryCheckpoint {
        QueryCheckpoint {
            checkpoint_id: id.to_owned(),
            source_volume_id: "volume-001".to_owned(),
            cumulative_edit_count: edit_count,
            graph_snapshot_sha256: snapshot.to_owned(),
            parent_graph_snapshot_sha256: parent.map(str::to_owned),
            edit_history_manifest_sha256: SHA_C.to_owned(),
            graph_builder_config_sha256: SHA_D.to_owned(),
            structural_uncertainty_manifest_sha256: SHA_B.to_owned(),
            scenario_results: scenarios,
        }
    }

    fn input(checkpoints: Vec<QueryCheckpoint>) -> QueryStabilityInput {
        QueryStabilityInput {
            query_id: "target-circuit-path-strength".to_owned(),
            query_definition_sha256: SHA_A.to_owned(),
            analysis_code_sha256: SHA_B.to_owned(),
            uncertainty_set_sha256: SHA_C.to_owned(),
            source_volume_id: "volume-001".to_owned(),
            metric_tolerances: vec![
                MetricTolerance {
                    metric_id: "global_motif_count".to_owned(),
                    absolute_tolerance: 0.05,
                    unit_id: "count".to_owned(),
                },
                MetricTolerance {
                    metric_id: "target_path_strength".to_owned(),
                    absolute_tolerance: 0.02,
                    unit_id: "normalized_strength".to_owned(),
                },
            ],
            checkpoints,
        }
    }

    #[test]
    fn stable_query_is_scoped_to_observed_window_and_does_not_claim_future() {
        let checkpoints = vec![
            checkpoint(
                "cp-1",
                10,
                SHA_A,
                None,
                vec![
                    scenario("h1", (10.0, 10.01), (0.70, 0.71)),
                    scenario("h2", (10.01, 10.02), (0.705, 0.715)),
                ],
            ),
            checkpoint(
                "cp-2",
                20,
                SHA_B,
                Some(SHA_A),
                vec![
                    scenario("h1", (10.01, 10.02), (0.705, 0.715)),
                    scenario("h2", (10.01, 10.03), (0.71, 0.72)),
                ],
            ),
        ];

        let report = audit_query_stability(&input(checkpoints)).unwrap();

        assert_eq!(report.status, QueryStabilityStatus::StableOverObservedWindow);
        assert!(!report.future_stability_certified);
        assert!(!report.universal_connectome_completeness_claimed);
        assert_eq!(report.metrics.len(), 2);
        assert!(report.metrics.iter().all(|metric| metric.stable_over_observed_window));
    }

    #[test]
    fn global_metric_can_stabilize_while_query_specific_edge_remains_unstable() {
        let checkpoints = vec![
            checkpoint(
                "cp-1",
                10,
                SHA_A,
                None,
                vec![scenario("h1", (10.0, 10.01), (0.20, 0.21))],
            ),
            checkpoint(
                "cp-2",
                20,
                SHA_B,
                Some(SHA_A),
                vec![scenario("h1", (10.0, 10.01), (0.80, 0.81))],
            ),
        ];

        let report = audit_query_stability(&input(checkpoints)).unwrap();

        assert_eq!(report.status, QueryStabilityStatus::UnstableOverObservedWindow);
        let global = report.metrics.iter().find(|metric| metric.metric_id == "global_motif_count").unwrap();
        let target = report.metrics.iter().find(|metric| metric.metric_id == "target_path_strength").unwrap();
        assert!(global.stable_over_observed_window);
        assert!(!target.stable_over_observed_window);
        assert!(target.worst_observed_temporal_drift_upper > 0.5);
    }

    #[test]
    fn one_checkpoint_is_insufficient_history_not_stable() {
        let report = audit_query_stability(&input(vec![checkpoint(
            "cp-1",
            10,
            SHA_A,
            None,
            vec![scenario("h1", (10.0, 10.01), (0.7, 0.71))],
        )]))
        .unwrap();

        assert_eq!(report.status, QueryStabilityStatus::InsufficientHistory);
        assert!(report.metrics.iter().all(|metric| !metric.stable_over_observed_window));
    }

    #[test]
    fn latest_structural_hypothesis_width_can_block_readiness() {
        let checkpoints = vec![
            checkpoint(
                "cp-1",
                10,
                SHA_A,
                None,
                vec![
                    scenario("h1", (10.0, 10.01), (0.70, 0.71)),
                    scenario("h2", (10.0, 10.01), (0.10, 0.90)),
                ],
            ),
            checkpoint(
                "cp-2",
                20,
                SHA_B,
                Some(SHA_A),
                vec![
                    scenario("h1", (10.0, 10.01), (0.70, 0.71)),
                    scenario("h2", (10.0, 10.01), (0.10, 0.90)),
                ],
            ),
        ];

        let report = audit_query_stability(&input(checkpoints)).unwrap();
        let target = report.metrics.iter().find(|metric| metric.metric_id == "target_path_strength").unwrap();

        assert!(!target.stable_over_observed_window);
        assert!((target.latest_identified_set_width - 0.8).abs() < 1e-12);
    }

    #[test]
    fn validates_parent_snapshot_chain() {
        let checkpoints = vec![
            checkpoint("cp-1", 10, SHA_A, None, vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
            checkpoint("cp-2", 20, SHA_B, Some(SHA_C), vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
        ];

        assert!(matches!(
            audit_query_stability(&input(checkpoints)),
            Err(StabilityError::ParentSnapshotMismatch { .. })
        ));
    }

    #[test]
    fn validates_strictly_increasing_edit_counts() {
        let checkpoints = vec![
            checkpoint("cp-1", 20, SHA_A, None, vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
            checkpoint("cp-2", 20, SHA_B, Some(SHA_A), vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
        ];

        assert_eq!(
            audit_query_stability(&input(checkpoints)),
            Err(StabilityError::NonMonotonicEditCount {
                previous: 20,
                current: 20,
            })
        );
    }

    #[test]
    fn validates_snapshot_and_query_provenance_digests() {
        let mut bad = input(vec![
            checkpoint("cp-1", 10, "not-a-sha", None, vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
            checkpoint("cp-2", 20, SHA_B, Some("not-a-sha"), vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
        ]);
        assert!(matches!(
            audit_query_stability(&bad),
            Err(StabilityError::InvalidSha256 { .. })
        ));

        bad = input(vec![checkpoint(
            "cp-1",
            10,
            SHA_A,
            None,
            vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))],
        )]);
        bad.analysis_code_sha256 = "short".to_owned();
        assert!(matches!(
            audit_query_stability(&bad),
            Err(StabilityError::InvalidSha256 { kind: "analysis code", .. })
        ));
    }

    #[test]
    fn rejects_missing_and_undeclared_query_metrics() {
        let mut checkpoints = vec![
            checkpoint("cp-1", 10, SHA_A, None, vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
            checkpoint("cp-2", 20, SHA_B, Some(SHA_A), vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
        ];
        checkpoints[1].scenario_results[0].metric_intervals.pop();
        assert!(matches!(
            audit_query_stability(&input(checkpoints.clone())),
            Err(StabilityError::MissingMetric { .. })
        ));

        checkpoints[1].scenario_results[0].metric_intervals.push(metric("unknown_metric", 0.0, 0.0));
        assert!(matches!(
            audit_query_stability(&input(checkpoints)),
            Err(StabilityError::UnexpectedMetric { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_scenarios_and_duplicate_metrics() {
        let mut checkpoints = vec![
            checkpoint("cp-1", 10, SHA_A, None, vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
            checkpoint("cp-2", 20, SHA_B, Some(SHA_A), vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
        ];
        checkpoints[1].scenario_results.push(scenario("h1", (1.0, 1.0), (1.0, 1.0)));
        assert!(matches!(
            audit_query_stability(&input(checkpoints.clone())),
            Err(StabilityError::DuplicateIdentifier { kind: "scenario", .. })
        ));

        checkpoints[1].scenario_results.truncate(1);
        checkpoints[1].scenario_results[0].metric_intervals.push(metric("global_motif_count", 1.0, 1.0));
        assert!(matches!(
            audit_query_stability(&input(checkpoints)),
            Err(StabilityError::DuplicateMetric { .. })
        ));
    }

    #[test]
    fn rejects_volume_mismatch_and_invalid_interval() {
        let mut checkpoints = vec![
            checkpoint("cp-1", 10, SHA_A, None, vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
            checkpoint("cp-2", 20, SHA_B, Some(SHA_A), vec![scenario("h1", (1.0, 1.0), (1.0, 1.0))]),
        ];
        checkpoints[1].source_volume_id = "other-volume".to_owned();
        assert!(matches!(
            audit_query_stability(&input(checkpoints.clone())),
            Err(StabilityError::SourceVolumeMismatch { .. })
        ));

        checkpoints[1].source_volume_id = "volume-001".to_owned();
        checkpoints[1].scenario_results[0].metric_intervals[0].bounds =
            NumericInterval { lower: 2.0, upper: 1.0 };
        assert!(matches!(
            audit_query_stability(&input(checkpoints)),
            Err(StabilityError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn preserves_scenario_intervals_and_does_not_average_them() {
        let checkpoints = vec![
            checkpoint(
                "cp-1",
                10,
                SHA_A,
                None,
                vec![
                    scenario("h1", (1.0, 1.0), (0.2, 0.3)),
                    scenario("h2", (1.0, 1.0), (0.7, 0.8)),
                ],
            ),
            checkpoint(
                "cp-2",
                20,
                SHA_B,
                Some(SHA_A),
                vec![
                    scenario("h1", (1.0, 1.0), (0.2, 0.3)),
                    scenario("h2", (1.0, 1.0), (0.7, 0.8)),
                ],
            ),
        ];

        let report = audit_query_stability(&input(checkpoints)).unwrap();
        let target = report.metrics.iter().find(|metric| metric.metric_id == "target_path_strength").unwrap();

        assert_eq!(target.latest_scenario_bounds.len(), 2);
        assert!((target.latest_identified_set_hull.lower - 0.2).abs() < 1e-12);
        assert!((target.latest_identified_set_hull.upper - 0.8).abs() < 1e-12);
        assert!(!target.stable_over_observed_window);
    }

    #[test]
    fn checkpoint_and_scenario_input_order_does_not_change_metric_audit() {
        let a = checkpoint(
            "cp-1",
            10,
            SHA_A,
            None,
            vec![
                scenario("h1", (1.0, 1.01), (0.4, 0.41)),
                scenario("h2", (1.0, 1.01), (0.405, 0.415)),
            ],
        );
        let b = checkpoint(
            "cp-2",
            20,
            SHA_B,
            Some(SHA_A),
            vec![
                scenario("h1", (1.0, 1.01), (0.405, 0.415)),
                scenario("h2", (1.0, 1.01), (0.41, 0.42)),
            ],
        );

        let first = audit_query_stability(&input(vec![a.clone(), b.clone()])).unwrap();
        let mut a_reversed = a;
        a_reversed.scenario_results.reverse();
        let mut b_reversed = b;
        b_reversed.scenario_results.reverse();
        let second = audit_query_stability(&input(vec![a_reversed, b_reversed])).unwrap();

        assert_eq!(first.metrics, second.metrics);
        assert_eq!(first.status, second.status);
    }
}
