// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-metric normalization for resource-planning shadow selectors.
//!
//! HDC and other bounded shadow selectors need comparable dimensionless inputs,
//! but normalization is itself a semantic decision. This crate therefore requires
//! one explicit ideal/worst band for every exact `ObjectiveMetric` and retains the
//! raw value, effective value, and whether explicit clamping occurred.
//!
//! No unit conversion, metric substitution, evidence qualification, ranking,
//! lease creation, or execution authority occurs here.

#![deny(unsafe_code)]

use std::collections::BTreeMap;
use symthaea_operations_research::ObjectiveDirection;
use symthaea_resource_objective_metric::{
    ObjectiveMetric, ObjectiveMetricError, ObjectiveMetricSchema, ObjectiveMetricSchemaError,
};
use symthaea_resource_objective_qualification::{
    QualifiedObjectiveValue, QualifiedObjectiveVector,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfRangePolicy {
    /// Values outside the declared ideal/worst interval fail normalization.
    Reject,
    /// Values outside the interval are explicitly clamped to the nearest endpoint.
    Clamp,
}

/// One exact metric's normalization theorem.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveNormalizationBand {
    pub metric: ObjectiveMetric,
    pub direction: ObjectiveDirection,
    pub ideal: f64,
    pub worst: f64,
    pub out_of_range: OutOfRangePolicy,
}

impl ObjectiveNormalizationBand {
    pub fn validate(&self) -> Result<(), NormalizationProfileError> {
        self.metric
            .validate()
            .map_err(NormalizationProfileError::InvalidMetric)?;
        if !self.ideal.is_finite() {
            return Err(NormalizationProfileError::NonFiniteIdeal {
                metric_id: self.metric.metric_id.clone(),
                value: self.ideal,
            });
        }
        if !self.worst.is_finite() {
            return Err(NormalizationProfileError::NonFiniteWorst {
                metric_id: self.metric.metric_id.clone(),
                value: self.worst,
            });
        }
        if self.ideal == self.worst {
            return Err(NormalizationProfileError::DegenerateBand {
                metric_id: self.metric.metric_id.clone(),
                value: self.ideal,
            });
        }
        match self.direction {
            ObjectiveDirection::Minimize if self.ideal >= self.worst => {
                return Err(NormalizationProfileError::DirectionInconsistent {
                    metric_id: self.metric.metric_id.clone(),
                    direction: "minimize",
                    ideal: self.ideal,
                    worst: self.worst,
                });
            }
            ObjectiveDirection::Maximize if self.ideal <= self.worst => {
                return Err(NormalizationProfileError::DirectionInconsistent {
                    metric_id: self.metric.metric_id.clone(),
                    direction: "maximize",
                    ideal: self.ideal,
                    worst: self.worst,
                });
            }
            _ => {}
        }
        Ok(())
    }

    pub fn lower_bound(&self) -> f64 {
        self.ideal.min(self.worst)
    }

    pub fn upper_bound(&self) -> f64 {
        self.ideal.max(self.worst)
    }
}

/// Canonical complete normalization policy for one exact metric schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveNormalizationProfile {
    metric_schema: ObjectiveMetricSchema,
    bands: BTreeMap<String, ObjectiveNormalizationBand>,
}

impl ObjectiveNormalizationProfile {
    pub fn new(
        bands: impl IntoIterator<Item = ObjectiveNormalizationBand>,
    ) -> Result<Self, NormalizationProfileError> {
        let mut collected = Vec::new();
        for band in bands {
            band.validate()?;
            collected.push(band);
        }
        if collected.is_empty() {
            return Err(NormalizationProfileError::EmptyProfile);
        }

        let metric_schema = ObjectiveMetricSchema::new(
            collected.iter().map(|band| band.metric.clone()),
        )
        .map_err(NormalizationProfileError::InvalidMetricSchema)?;

        let mut by_name = BTreeMap::new();
        for band in collected {
            let previous = by_name.insert(band.metric.objective_name.clone(), band);
            debug_assert!(previous.is_none());
        }

        Ok(Self {
            metric_schema,
            bands: by_name,
        })
    }

    pub fn metric_schema(&self) -> &ObjectiveMetricSchema {
        &self.metric_schema
    }

    pub fn get(&self, objective_name: &str) -> Option<&ObjectiveNormalizationBand> {
        self.bands.get(objective_name)
    }

    pub fn bands(&self) -> impl Iterator<Item = &ObjectiveNormalizationBand> {
        self.bands.values()
    }

    pub fn len(&self) -> usize {
        self.bands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }
}

/// Positive normalized scalar retaining both raw and effective values.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedObjectiveValue {
    metric: ObjectiveMetric,
    raw_value: f64,
    effective_value: f64,
    quality: f64,
    clamped: bool,
}

impl NormalizedObjectiveValue {
    pub fn metric(&self) -> &ObjectiveMetric {
        &self.metric
    }

    pub fn raw_value(&self) -> f64 {
        self.raw_value
    }

    pub fn effective_value(&self) -> f64 {
        self.effective_value
    }

    /// Dimensionless quality in [0, 1], where ideal = 1 and worst = 0.
    pub fn quality(&self) -> f64 {
        self.quality
    }

    pub fn clamped(&self) -> bool {
        self.clamped
    }
}

/// Complete normalized vector retaining the exact profile that interpreted it.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedObjectiveVector {
    candidate_id: String,
    profile: ObjectiveNormalizationProfile,
    values: BTreeMap<String, NormalizedObjectiveValue>,
}

impl NormalizedObjectiveVector {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn profile(&self) -> &ObjectiveNormalizationProfile {
        &self.profile
    }

    pub fn values(&self) -> &BTreeMap<String, NormalizedObjectiveValue> {
        &self.values
    }

    pub fn get(&self, objective_name: &str) -> Option<&NormalizedObjectiveValue> {
        self.values.get(objective_name)
    }
}

/// Normalize one already-qualified scalar under one exact semantic band.
pub fn normalize_objective_value(
    qualified: &QualifiedObjectiveValue,
    band: &ObjectiveNormalizationBand,
) -> Result<NormalizedObjectiveValue, NormalizationError> {
    band.validate().map_err(NormalizationError::InvalidProfile)?;
    if qualified.metric() != &band.metric {
        return Err(NormalizationError::MetricMismatch {
            objective_name: band.metric.objective_name.clone(),
            expected_metric_id: band.metric.metric_id.clone(),
            actual_metric_id: qualified.metric().metric_id.clone(),
        });
    }

    let raw = qualified.value();
    if !raw.is_finite() {
        return Err(NormalizationError::NonFiniteQualifiedValue {
            metric_id: band.metric.metric_id.clone(),
            value: raw,
        });
    }

    let lower = band.lower_bound();
    let upper = band.upper_bound();
    let out_of_range = raw < lower || raw > upper;
    let (effective, clamped) = match (out_of_range, band.out_of_range) {
        (false, _) => (raw, false),
        (true, OutOfRangePolicy::Reject) => {
            return Err(NormalizationError::OutOfRange {
                metric_id: band.metric.metric_id.clone(),
                value: raw,
                lower,
                upper,
            });
        }
        (true, OutOfRangePolicy::Clamp) => (raw.clamp(lower, upper), true),
    };

    // This single formula works for both directions because validation requires
    // the ideal/worst ordering to follow the declared optimization direction.
    let quality = (effective - band.worst) / (band.ideal - band.worst);
    if !quality.is_finite() {
        return Err(NormalizationError::NonFiniteNormalizedQuality {
            metric_id: band.metric.metric_id.clone(),
            value: quality,
        });
    }

    Ok(NormalizedObjectiveValue {
        metric: band.metric.clone(),
        raw_value: raw,
        effective_value: effective,
        quality: quality.clamp(0.0, 1.0),
        clamped,
    })
}

/// Normalize a complete qualified vector under an exactly matching metric schema.
/// No partial positive vector is returned.
pub fn normalize_objective_vector(
    qualified: &QualifiedObjectiveVector,
    profile: &ObjectiveNormalizationProfile,
) -> Result<NormalizedObjectiveVector, NormalizationError> {
    if qualified.metric_schema() != profile.metric_schema() {
        return Err(NormalizationError::MetricSchemaMismatch);
    }

    let mut values = BTreeMap::new();
    for band in profile.bands() {
        let objective_name = &band.metric.objective_name;
        let scalar = qualified
            .get(objective_name)
            .ok_or_else(|| NormalizationError::QualifiedObjectiveMissing {
                objective_name: objective_name.clone(),
            })?;
        let normalized = normalize_objective_value(scalar, band)?;
        values.insert(objective_name.clone(), normalized);
    }

    Ok(NormalizedObjectiveVector {
        candidate_id: qualified.candidate_id().to_owned(),
        profile: profile.clone(),
        values,
    })
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum NormalizationProfileError {
    #[error("normalization profile must contain at least one band")]
    EmptyProfile,
    #[error("invalid objective metric: {0}")]
    InvalidMetric(ObjectiveMetricError),
    #[error("invalid objective metric schema: {0}")]
    InvalidMetricSchema(ObjectiveMetricSchemaError),
    #[error("ideal for {metric_id} is non-finite: {value}")]
    NonFiniteIdeal { metric_id: String, value: f64 },
    #[error("worst for {metric_id} is non-finite: {value}")]
    NonFiniteWorst { metric_id: String, value: f64 },
    #[error("ideal and worst for {metric_id} are both {value}")]
    DegenerateBand { metric_id: String, value: f64 },
    #[error("{metric_id} normalization is inconsistent with {direction}: ideal={ideal}, worst={worst}")]
    DirectionInconsistent {
        metric_id: String,
        direction: &'static str,
        ideal: f64,
        worst: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum NormalizationError {
    #[error("invalid normalization profile: {0}")]
    InvalidProfile(NormalizationProfileError),
    #[error("qualified metric schema differs from normalization profile")]
    MetricSchemaMismatch,
    #[error("qualified vector is missing objective {objective_name}")]
    QualifiedObjectiveMissing { objective_name: String },
    #[error("metric mismatch for {objective_name}: expected {expected_metric_id}, got {actual_metric_id}")]
    MetricMismatch {
        objective_name: String,
        expected_metric_id: String,
        actual_metric_id: String,
    },
    #[error("qualified value for {metric_id} is non-finite: {value}")]
    NonFiniteQualifiedValue { metric_id: String, value: f64 },
    #[error("value {value} for {metric_id} is outside [{lower}, {upper}]")]
    OutOfRange {
        metric_id: String,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("normalized quality for {metric_id} is non-finite: {value}")]
    NonFiniteNormalizedQuality { metric_id: String, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_feasible_set::enumerate_feasible_set;
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_objective_evidence::{
        CandidateObjectiveEvidence, ObjectiveEvidenceClass, ObjectiveEvidenceScope,
        ResourceObjectiveEvidenceSet,
    };
    use symthaea_resource_objective_metric::{ObjectiveMetric, ObjectiveStatistic};
    use symthaea_resource_objective_qualification::{
        qualify_objective_vector, ObjectiveEvidenceIdentityPolicy,
        ObjectiveMultipleEvidenceRule, ObjectiveQualificationPolicy,
        ObjectiveQualificationRequest,
    };
    use symthaea_resource_quality::{
        NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
        ResourceQualityRequirement,
    };
    use symthaea_resource_quality_evidence::{
        QualityEvidenceClass, QualityEvidenceSet, QualityEvidenceWindow, QualitySubject,
    };
    use symthaea_resource_quality_qualification::{
        EvidenceIdentityPolicy, MultipleEvidenceRule, QualityQualificationPolicy,
    };
    use symthaea_resource_topology::{ResourceLink, ResourceTopology};

    fn t0() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn energy() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.joule.v1",
            "si.joule.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn resilience() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "resilience",
            "resilience.fraction.v1",
            "ratio.fraction.v1",
            ObjectiveStatistic::CandidateFraction,
        )
        .unwrap()
    }

    fn energy_kwh() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.kwh.v1",
            "energy.kilowatt_hour.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn qualified_vector(
        energy_value: f64,
        resilience_value: f64,
    ) -> QualifiedObjectiveVector {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [ResourcePort {
                    id: "out".into(),
                    direction: PortDirection::Output,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [ResourcePort {
                    id: "in".into(),
                    direction: PortDirection::Input,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: power(100.0),
                loss_fraction: 0.0,
            })
            .unwrap();

        let mut capacities = CapacitySchedule::default();
        for (id, subject) in [
            (
                "source-cap",
                CapacitySubject::Port {
                    node_id: "source".into(),
                    port_id: "out".into(),
                },
            ),
            (
                "link-cap",
                CapacitySubject::Link {
                    link_id: "line".into(),
                },
            ),
            (
                "sink-cap",
                CapacitySubject::Port {
                    node_id: "sink".into(),
                    port_id: "in".into(),
                },
            ),
        ] {
            capacities
                .add_window(
                    &topology,
                    CapacityWindow {
                        id: id.into(),
                        subject,
                        valid_from: t0(),
                        valid_until: t0() + Duration::hours(1),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }

        let mut quality_profile = ResourceQualityProfile::new(power(1.0).key);
        quality_profile
            .set_numeric(QualityMetric::TemperatureCelsius, 40.0)
            .unwrap();
        let mut quality_evidence = QualityEvidenceSet::default();
        quality_evidence
            .insert(
                &topology,
                QualityEvidenceWindow {
                    id: "quality".into(),
                    subject: QualitySubject::Link {
                        link_id: "line".into(),
                    },
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    profile: quality_profile,
                    evidence_class: QualityEvidenceClass::Observed,
                    evidence_ref: "sensor:quality".into(),
                },
            )
            .unwrap();
        let mut quality_requirement = ResourceQualityRequirement::new(power(1.0).key);
        quality_requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(20.0),
                maximum: Some(60.0),
            })
            .unwrap();
        let quality_policy = QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        };
        let feasible_set = enumerate_feasible_set(
            &topology,
            &capacities,
            &AllocationBook::default(),
            &quality_evidence,
            &quality_requirement,
            &quality_policy,
            vec![PlannedAllocation {
                id: "candidate".into(),
                link_id: "line".into(),
                valid_from: t0(),
                valid_until: t0() + Duration::hours(1),
                sent: power(40.0),
            }],
            8,
        )
        .unwrap();

        let mut evidence = ResourceObjectiveEvidenceSet::new(feasible_set);
        for (id, metric, value) in [
            ("energy", energy(), energy_value),
            ("resilience", resilience(), resilience_value),
        ] {
            evidence
                .insert(CandidateObjectiveEvidence {
                    id: id.into(),
                    candidate_id: "candidate".into(),
                    metric,
                    value,
                    evidence_class: ObjectiveEvidenceClass::Observed,
                    evidence_ref: format!("source:{id}"),
                    scope: ObjectiveEvidenceScope::CandidateAggregate,
                })
                .unwrap();
        }

        let policy = ObjectiveQualificationPolicy {
            allowed_classes: [ObjectiveEvidenceClass::Observed].into_iter().collect(),
            identity_policy: ObjectiveEvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: ObjectiveMultipleEvidenceRule::RejectMultiple,
        };
        qualify_objective_vector(
            &evidence,
            "candidate",
            &[
                ObjectiveQualificationRequest {
                    metric: resilience(),
                    policy: policy.clone(),
                },
                ObjectiveQualificationRequest {
                    metric: energy(),
                    policy,
                },
            ],
        )
        .unwrap()
    }

    fn profile(policy: OutOfRangePolicy) -> ObjectiveNormalizationProfile {
        ObjectiveNormalizationProfile::new([
            ObjectiveNormalizationBand {
                metric: energy(),
                direction: ObjectiveDirection::Minimize,
                ideal: 0.0,
                worst: 100.0,
                out_of_range: policy,
            },
            ObjectiveNormalizationBand {
                metric: resilience(),
                direction: ObjectiveDirection::Maximize,
                ideal: 1.0,
                worst: 0.0,
                out_of_range: policy,
            },
        ])
        .unwrap()
    }

    #[test]
    fn minimize_and_maximize_metrics_share_ideal_one_worst_zero_semantics() {
        let normalized = normalize_objective_vector(&qualified_vector(25.0, 0.8), &profile(OutOfRangePolicy::Reject)).unwrap();
        assert!((normalized.get("energy").unwrap().quality() - 0.75).abs() < 1e-12);
        assert!((normalized.get("resilience").unwrap().quality() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn reject_policy_fails_out_of_range_instead_of_silently_clamping() {
        let error = normalize_objective_vector(
            &qualified_vector(125.0, 0.8),
            &profile(OutOfRangePolicy::Reject),
        )
        .unwrap_err();
        assert!(matches!(error, NormalizationError::OutOfRange { metric_id, .. } if metric_id == "energy.total.joule.v1"));
    }

    #[test]
    fn clamp_policy_is_explicit_and_preserves_raw_value() {
        let normalized = normalize_objective_vector(
            &qualified_vector(-25.0, 1.2),
            &profile(OutOfRangePolicy::Clamp),
        )
        .unwrap();
        let energy_value = normalized.get("energy").unwrap();
        assert_eq!(energy_value.raw_value(), -25.0);
        assert_eq!(energy_value.effective_value(), 0.0);
        assert_eq!(energy_value.quality(), 1.0);
        assert!(energy_value.clamped());
        let resilience_value = normalized.get("resilience").unwrap();
        assert_eq!(resilience_value.raw_value(), 1.2);
        assert_eq!(resilience_value.effective_value(), 1.0);
        assert_eq!(resilience_value.quality(), 1.0);
        assert!(resilience_value.clamped());
    }

    #[test]
    fn direction_inconsistent_bands_fail_closed() {
        assert!(matches!(
            ObjectiveNormalizationProfile::new([ObjectiveNormalizationBand {
                metric: energy(),
                direction: ObjectiveDirection::Minimize,
                ideal: 100.0,
                worst: 0.0,
                out_of_range: OutOfRangePolicy::Reject,
            }]),
            Err(NormalizationProfileError::DirectionInconsistent { .. })
        ));
        assert!(matches!(
            ObjectiveNormalizationProfile::new([ObjectiveNormalizationBand {
                metric: resilience(),
                direction: ObjectiveDirection::Maximize,
                ideal: 0.0,
                worst: 1.0,
                out_of_range: OutOfRangePolicy::Reject,
            }]),
            Err(NormalizationProfileError::DirectionInconsistent { .. })
        ));
    }

    #[test]
    fn same_name_different_metric_schema_cannot_normalize_vector() {
        let mismatched = ObjectiveNormalizationProfile::new([
            ObjectiveNormalizationBand {
                metric: energy_kwh(),
                direction: ObjectiveDirection::Minimize,
                ideal: 0.0,
                worst: 1.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
            ObjectiveNormalizationBand {
                metric: resilience(),
                direction: ObjectiveDirection::Maximize,
                ideal: 1.0,
                worst: 0.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
        ])
        .unwrap();
        assert!(matches!(
            normalize_objective_vector(&qualified_vector(25.0, 0.8), &mismatched),
            Err(NormalizationError::MetricSchemaMismatch)
        ));
    }

    #[test]
    fn duplicate_objective_names_fail_profile_schema() {
        assert!(matches!(
            ObjectiveNormalizationProfile::new([
                ObjectiveNormalizationBand {
                    metric: energy(),
                    direction: ObjectiveDirection::Minimize,
                    ideal: 0.0,
                    worst: 100.0,
                    out_of_range: OutOfRangePolicy::Reject,
                },
                ObjectiveNormalizationBand {
                    metric: energy_kwh(),
                    direction: ObjectiveDirection::Minimize,
                    ideal: 0.0,
                    worst: 1.0,
                    out_of_range: OutOfRangePolicy::Reject,
                },
            ]),
            Err(NormalizationProfileError::InvalidMetricSchema(
                ObjectiveMetricSchemaError::DuplicateObjectiveName(name)
            )) if name == "energy"
        ));
    }
}
