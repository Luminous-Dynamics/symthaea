// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact semantic identity for resource-planning objectives.
//!
//! A name such as `energy` or `latency` is not enough to make two scalar values
//! comparable. This crate binds every rankable objective to an exact metric id,
//! unit id, and aggregate statistic before evidence qualification or optimization.
//!
//! Examples:
//! - `energy.total.joule.v1` / `si.joule.v1` / `CandidateTotal`
//! - `latency.p95.second.v1` / `si.second.v1` / `PercentileBasisPoints(9500)`
//! - `availability.fraction.v1` / `ratio.fraction.v1` / `CandidateFraction`
//!
//! This crate does not measure a metric, authenticate evidence, rank candidates,
//! or define institutional/execution authority.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Exact aggregate/statistical meaning of one candidate-level scalar.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectiveStatistic {
    /// Sum/integral over the complete candidate plan.
    CandidateTotal,
    /// Arithmetic/time-normalized mean over the complete candidate plan.
    CandidateMean,
    /// Minimum value observed/estimated over the candidate plan.
    CandidateMinimum,
    /// Maximum value observed/estimated over the candidate plan.
    CandidateMaximum,
    /// Value at the candidate plan's terminal boundary.
    CandidateFinal,
    /// Count of events/items over the candidate plan.
    CandidateCount,
    /// Dimensionless fraction over the candidate plan.
    CandidateFraction,
    /// Candidate-level percentile in basis points, where 9500 means p95.
    PercentileBasisPoints(u16),
}

impl ObjectiveStatistic {
    pub fn validate(&self) -> Result<(), ObjectiveMetricError> {
        if let Self::PercentileBasisPoints(value) = self
            && *value > 10_000
        {
            return Err(ObjectiveMetricError::InvalidPercentileBasisPoints(*value));
        }
        Ok(())
    }
}

/// Exact metric semantics used to interpret one scalar objective value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectiveMetric {
    /// Human-facing objective role, e.g. `energy` or `tail_latency`.
    pub objective_name: String,
    /// Stable versioned semantic metric identifier.
    pub metric_id: String,
    /// Stable canonical unit identifier.
    pub unit_id: String,
    /// Exact aggregate/statistical interpretation.
    pub statistic: ObjectiveStatistic,
}

impl ObjectiveMetric {
    pub fn new(
        objective_name: impl Into<String>,
        metric_id: impl Into<String>,
        unit_id: impl Into<String>,
        statistic: ObjectiveStatistic,
    ) -> Result<Self, ObjectiveMetricError> {
        let metric = Self {
            objective_name: objective_name.into(),
            metric_id: metric_id.into(),
            unit_id: unit_id.into(),
            statistic,
        };
        metric.validate()?;
        Ok(metric)
    }

    pub fn validate(&self) -> Result<(), ObjectiveMetricError> {
        if self.objective_name.trim().is_empty() {
            return Err(ObjectiveMetricError::BlankObjectiveName);
        }
        if self.metric_id.trim().is_empty() {
            return Err(ObjectiveMetricError::BlankMetricId);
        }
        if self.unit_id.trim().is_empty() {
            return Err(ObjectiveMetricError::BlankUnitId);
        }
        self.statistic.validate()?;
        Ok(())
    }

    /// Exact semantic compatibility. No implicit unit conversion, aliasing, or
    /// statistic substitution is performed at this boundary.
    pub fn is_exactly_compatible_with(&self, other: &Self) -> bool {
        self == other
    }
}

/// A validated named metric schema for one planner/ranking context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveMetricSchema {
    metrics: BTreeMap<String, ObjectiveMetric>,
}

impl ObjectiveMetricSchema {
    pub fn new(
        metrics: impl IntoIterator<Item = ObjectiveMetric>,
    ) -> Result<Self, ObjectiveMetricSchemaError> {
        let mut by_name = BTreeMap::new();
        let mut metric_ids = BTreeSet::new();
        for metric in metrics {
            metric
                .validate()
                .map_err(ObjectiveMetricSchemaError::InvalidMetric)?;
            if by_name.contains_key(&metric.objective_name) {
                return Err(ObjectiveMetricSchemaError::DuplicateObjectiveName(
                    metric.objective_name,
                ));
            }
            if !metric_ids.insert(metric.metric_id.clone()) {
                return Err(ObjectiveMetricSchemaError::DuplicateMetricId(
                    metric.metric_id,
                ));
            }
            by_name.insert(metric.objective_name.clone(), metric);
        }
        if by_name.is_empty() {
            return Err(ObjectiveMetricSchemaError::EmptySchema);
        }
        Ok(Self { metrics: by_name })
    }

    pub fn get(&self, objective_name: &str) -> Option<&ObjectiveMetric> {
        self.metrics.get(objective_name)
    }

    pub fn metrics(&self) -> impl Iterator<Item = &ObjectiveMetric> {
        self.metrics.values()
    }

    pub fn objective_names(&self) -> impl Iterator<Item = &str> {
        self.metrics.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ObjectiveMetricError {
    #[error("objective name must not be blank")]
    BlankObjectiveName,
    #[error("metric id must not be blank")]
    BlankMetricId,
    #[error("unit id must not be blank")]
    BlankUnitId,
    #[error("percentile basis points {0} exceeds 10000")]
    InvalidPercentileBasisPoints(u16),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ObjectiveMetricSchemaError {
    #[error("objective metric schema must contain at least one metric")]
    EmptySchema,
    #[error("invalid objective metric: {0}")]
    InvalidMetric(ObjectiveMetricError),
    #[error("duplicate objective name {0}")]
    DuplicateObjectiveName(String),
    #[error("duplicate metric id {0}")]
    DuplicateMetricId(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.joule.v1",
            "si.joule.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn latency_p95() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "tail_latency",
            "latency.p95.second.v1",
            "si.second.v1",
            ObjectiveStatistic::PercentileBasisPoints(9500),
        )
        .unwrap()
    }

    #[test]
    fn exact_metric_identity_contains_unit_and_statistic() {
        let metric = energy();
        assert_eq!(metric.objective_name, "energy");
        assert_eq!(metric.metric_id, "energy.total.joule.v1");
        assert_eq!(metric.unit_id, "si.joule.v1");
        assert_eq!(metric.statistic, ObjectiveStatistic::CandidateTotal);
    }

    #[test]
    fn same_name_different_unit_is_not_compatible() {
        let joule = energy();
        let kilowatt_hour = ObjectiveMetric::new(
            "energy",
            "energy.total.kwh.v1",
            "energy.kilowatt_hour.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap();
        assert!(!joule.is_exactly_compatible_with(&kilowatt_hour));
    }

    #[test]
    fn same_name_and_unit_different_statistic_is_not_compatible() {
        let p95 = latency_p95();
        let mean = ObjectiveMetric::new(
            "tail_latency",
            "latency.mean.second.v1",
            "si.second.v1",
            ObjectiveStatistic::CandidateMean,
        )
        .unwrap();
        assert!(!p95.is_exactly_compatible_with(&mean));
    }

    #[test]
    fn invalid_percentile_fails_closed() {
        assert!(matches!(
            ObjectiveMetric::new(
                "latency",
                "latency.invalid.v1",
                "si.second.v1",
                ObjectiveStatistic::PercentileBasisPoints(10_001),
            ),
            Err(ObjectiveMetricError::InvalidPercentileBasisPoints(10_001))
        ));
    }

    #[test]
    fn schema_is_canonical_by_objective_name() {
        let schema = ObjectiveMetricSchema::new([latency_p95(), energy()]).unwrap();
        assert_eq!(
            schema.objective_names().collect::<Vec<_>>(),
            vec!["energy", "tail_latency"]
        );
    }

    #[test]
    fn schema_rejects_duplicate_objective_names() {
        let a = energy();
        let b = ObjectiveMetric::new(
            "energy",
            "energy.mean.watt.v1",
            "si.watt.v1",
            ObjectiveStatistic::CandidateMean,
        )
        .unwrap();
        assert!(matches!(
            ObjectiveMetricSchema::new([a, b]),
            Err(ObjectiveMetricSchemaError::DuplicateObjectiveName(name)) if name == "energy"
        ));
    }

    #[test]
    fn schema_rejects_reused_metric_identity_under_an_alias() {
        let a = energy();
        let b = ObjectiveMetric::new(
            "energy_alias",
            "energy.total.joule.v1",
            "si.joule.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap();
        assert!(matches!(
            ObjectiveMetricSchema::new([a, b]),
            Err(ObjectiveMetricSchemaError::DuplicateMetricId(id)) if id == "energy.total.joule.v1"
        ));
    }

    #[test]
    fn blank_semantics_fail_closed() {
        assert!(matches!(
            ObjectiveMetric::new(
                " ",
                "metric.v1",
                "si.joule.v1",
                ObjectiveStatistic::CandidateTotal,
            ),
            Err(ObjectiveMetricError::BlankObjectiveName)
        ));
        assert!(matches!(
            ObjectiveMetric::new(
                "energy",
                " ",
                "si.joule.v1",
                ObjectiveStatistic::CandidateTotal,
            ),
            Err(ObjectiveMetricError::BlankMetricId)
        ));
        assert!(matches!(
            ObjectiveMetric::new(
                "energy",
                "energy.total.joule.v1",
                " ",
                ObjectiveStatistic::CandidateTotal,
            ),
            Err(ObjectiveMetricError::BlankUnitId)
        ));
    }
}
