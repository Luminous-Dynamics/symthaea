// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded parent-to-child operating authority for multiscale infrastructure.
//!
//! Higher-scale nodes may constrain lower-scale nodes, but this crate deliberately
//! does not grant arbitrary actuator access. An operating envelope is a finite,
//! expiring set of admissible modes and observable resource bounds. Evaluation is
//! fail-closed on expiry, missing observations, invalid metrics, or invalid authority.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_resource_hierarchy::ResourceHierarchy;
use symthaea_resource_model::ResourceKey;
use thiserror::Error;

/// Externally observable aspect of a resource boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BoundaryMetric {
    Import,
    Export,
    Reserve,
    AvailableCapacity,
    Demand,
}

/// Generic operating posture shared across infrastructure domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperatingMode {
    Normal,
    Degraded,
    Islanded,
    Emergency,
    Maintenance,
}

/// Fraction-valued operating observation used in diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FractionMetric {
    CriticalServiceFraction,
    ShedFraction,
}

/// Bounded constraint on one observed resource metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceConstraint {
    pub key: ResourceKey,
    pub metric: BoundaryMetric,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

impl ResourceConstraint {
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        if self.minimum.is_none() && self.maximum.is_none() {
            return Err(EnvelopeError::UnboundedConstraint);
        }
        for value in [self.minimum, self.maximum].into_iter().flatten() {
            if !value.is_finite() || value < 0.0 {
                return Err(EnvelopeError::InvalidNonNegativeValue(value));
            }
        }
        if let (Some(minimum), Some(maximum)) = (self.minimum, self.maximum)
            && minimum > maximum
        {
            return Err(EnvelopeError::InvertedConstraint { minimum, maximum });
        }
        Ok(())
    }
}

/// One measured boundary metric used to evaluate an operating point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservedResourceMetric {
    pub key: ResourceKey,
    pub metric: BoundaryMetric,
    pub value: f64,
}

/// Current locally observed operating state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatingPoint {
    pub mode: OperatingMode,
    pub critical_service_fraction: f64,
    pub shed_fraction: f64,
    pub resources: Vec<ObservedResourceMetric>,
}

impl OperatingPoint {
    fn find(&self, key: ResourceKey, metric: BoundaryMetric) -> Option<f64> {
        self.resources
            .iter()
            .find(|observation| observation.key == key && observation.metric == metric)
            .map(|observation| observation.value)
    }
}

/// Expiring authority projected from a higher-scale node to a descendant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatingEnvelope {
    pub id: String,
    pub issuer_node_id: String,
    pub subject_node_id: String,
    /// Monotone policy generation supplied by the authority layer.
    pub generation: u64,
    pub valid_from: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
    pub allowed_modes: BTreeSet<OperatingMode>,
    pub resource_constraints: Vec<ResourceConstraint>,
    pub critical_service_floor: f64,
    pub max_shed_fraction: f64,
}

impl OperatingEnvelope {
    pub fn validate_structure(&self) -> Result<(), EnvelopeError> {
        if self.valid_until <= self.valid_from {
            return Err(EnvelopeError::InvalidValidityWindow);
        }
        if self.allowed_modes.is_empty() {
            return Err(EnvelopeError::NoAllowedModes);
        }
        validate_fraction("critical_service_floor", self.critical_service_floor)?;
        validate_fraction("max_shed_fraction", self.max_shed_fraction)?;
        for constraint in &self.resource_constraints {
            constraint.validate()?;
        }
        Ok(())
    }

    /// Verify that the issuer is a strict ancestor of the subject.
    pub fn validate_authority(&self, hierarchy: &ResourceHierarchy) -> Result<(), EnvelopeError> {
        if hierarchy.node(&self.issuer_node_id).is_none() {
            return Err(EnvelopeError::UnknownIssuer(self.issuer_node_id.clone()));
        }
        if hierarchy.node(&self.subject_node_id).is_none() {
            return Err(EnvelopeError::UnknownSubject(self.subject_node_id.clone()));
        }
        if self.issuer_node_id == self.subject_node_id {
            return Err(EnvelopeError::IssuerMustBeStrictAncestor);
        }

        let mut cursor = hierarchy.parent(&self.subject_node_id);
        while let Some(parent_id) = cursor {
            if parent_id == self.issuer_node_id {
                return Ok(());
            }
            cursor = hierarchy.parent(parent_id);
        }
        Err(EnvelopeError::IssuerNotAncestor {
            issuer: self.issuer_node_id.clone(),
            subject: self.subject_node_id.clone(),
        })
    }

    /// Evaluate a local operating point against this envelope.
    ///
    /// This method returns a complete violation set rather than stopping at the
    /// first failure, but any violation means the operating point is not compliant.
    pub fn evaluate(
        &self,
        hierarchy: &ResourceHierarchy,
        point: &OperatingPoint,
        now: DateTime<Utc>,
    ) -> Result<EnvelopeEvaluation, EnvelopeError> {
        self.validate_structure()?;
        self.validate_authority(hierarchy)?;

        let mut violations = Vec::new();
        if now < self.valid_from {
            violations.push(EnvelopeViolation::NotYetValid);
        }
        if now >= self.valid_until {
            violations.push(EnvelopeViolation::Expired);
        }
        if !self.allowed_modes.contains(&point.mode) {
            violations.push(EnvelopeViolation::ModeNotAllowed(point.mode));
        }
        if !point.critical_service_fraction.is_finite()
            || !(0.0..=1.0).contains(&point.critical_service_fraction)
        {
            violations.push(EnvelopeViolation::InvalidObservedFraction {
                metric: FractionMetric::CriticalServiceFraction,
                value: point.critical_service_fraction,
            });
        } else if point.critical_service_fraction < self.critical_service_floor {
            violations.push(EnvelopeViolation::CriticalServiceBelowFloor {
                observed: point.critical_service_fraction,
                floor: self.critical_service_floor,
            });
        }
        if !point.shed_fraction.is_finite() || !(0.0..=1.0).contains(&point.shed_fraction) {
            violations.push(EnvelopeViolation::InvalidObservedFraction {
                metric: FractionMetric::ShedFraction,
                value: point.shed_fraction,
            });
        } else if point.shed_fraction > self.max_shed_fraction {
            violations.push(EnvelopeViolation::ShedFractionAboveMaximum {
                observed: point.shed_fraction,
                maximum: self.max_shed_fraction,
            });
        }

        for constraint in &self.resource_constraints {
            let Some(observed) = point.find(constraint.key, constraint.metric) else {
                violations.push(EnvelopeViolation::MissingObservation {
                    key: constraint.key,
                    metric: constraint.metric,
                });
                continue;
            };
            if !observed.is_finite() || observed < 0.0 {
                violations.push(EnvelopeViolation::InvalidResourceObservation {
                    key: constraint.key,
                    metric: constraint.metric,
                    value: observed,
                });
                continue;
            }
            if let Some(minimum) = constraint.minimum
                && observed < minimum
            {
                violations.push(EnvelopeViolation::BelowMinimum {
                    key: constraint.key,
                    metric: constraint.metric,
                    observed,
                    minimum,
                });
            }
            if let Some(maximum) = constraint.maximum
                && observed > maximum
            {
                violations.push(EnvelopeViolation::AboveMaximum {
                    key: constraint.key,
                    metric: constraint.metric,
                    observed,
                    maximum,
                });
            }
        }

        Ok(EnvelopeEvaluation {
            compliant: violations.is_empty(),
            violations,
        })
    }
}

fn validate_fraction(label: &'static str, value: f64) -> Result<(), EnvelopeError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EnvelopeError::InvalidFraction { label, value });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeEvaluation {
    pub compliant: bool,
    pub violations: Vec<EnvelopeViolation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeViolation {
    NotYetValid,
    Expired,
    ModeNotAllowed(OperatingMode),
    CriticalServiceBelowFloor {
        observed: f64,
        floor: f64,
    },
    ShedFractionAboveMaximum {
        observed: f64,
        maximum: f64,
    },
    MissingObservation {
        key: ResourceKey,
        metric: BoundaryMetric,
    },
    InvalidObservedFraction {
        metric: FractionMetric,
        value: f64,
    },
    InvalidResourceObservation {
        key: ResourceKey,
        metric: BoundaryMetric,
        value: f64,
    },
    BelowMinimum {
        key: ResourceKey,
        metric: BoundaryMetric,
        observed: f64,
        minimum: f64,
    },
    AboveMaximum {
        key: ResourceKey,
        metric: BoundaryMetric,
        observed: f64,
        maximum: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum EnvelopeError {
    #[error("resource constraint has neither a minimum nor maximum")]
    UnboundedConstraint,
    #[error("expected a finite non-negative value, got {0}")]
    InvalidNonNegativeValue(f64),
    #[error("constraint minimum {minimum} exceeds maximum {maximum}")]
    InvertedConstraint { minimum: f64, maximum: f64 },
    #[error("invalid validity window")]
    InvalidValidityWindow,
    #[error("operating envelope must allow at least one mode")]
    NoAllowedModes,
    #[error("invalid fraction {label}: {value}")]
    InvalidFraction { label: &'static str, value: f64 },
    #[error("unknown issuer node {0}")]
    UnknownIssuer(String),
    #[error("unknown subject node {0}")]
    UnknownSubject(String),
    #[error("issuer must be a strict ancestor of the subject")]
    IssuerMustBeStrictAncestor,
    #[error("issuer {issuer} is not an ancestor of subject {subject}")]
    IssuerNotAncestor { issuer: String, subject: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::{ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit};

    fn watts_key() -> ResourceKey {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "region",
                "Region",
                NodeScale::Region,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "region",
                ResourceNode::new(
                    "site",
                    "Site",
                    NodeScale::Site,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-a",
                    "Rack A",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack-b",
                    "Rack B",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn envelope(issuer: &str, subject: &str) -> OperatingEnvelope {
        let mut allowed_modes = BTreeSet::new();
        allowed_modes.insert(OperatingMode::Normal);
        allowed_modes.insert(OperatingMode::Islanded);
        OperatingEnvelope {
            id: "env-1".into(),
            issuer_node_id: issuer.into(),
            subject_node_id: subject.into(),
            generation: 1,
            valid_from: t0(),
            valid_until: t0() + Duration::hours(1),
            allowed_modes,
            resource_constraints: vec![ResourceConstraint {
                key: watts_key(),
                metric: BoundaryMetric::Import,
                minimum: None,
                maximum: Some(100.0),
            }],
            critical_service_floor: 0.8,
            max_shed_fraction: 0.2,
        }
    }

    fn point() -> OperatingPoint {
        OperatingPoint {
            mode: OperatingMode::Normal,
            critical_service_fraction: 0.95,
            shed_fraction: 0.05,
            resources: vec![ObservedResourceMetric {
                key: watts_key(),
                metric: BoundaryMetric::Import,
                value: 80.0,
            }],
        }
    }

    #[test]
    fn strict_ancestor_can_issue_envelope() {
        let hierarchy = hierarchy();
        assert!(envelope("site", "rack-a")
            .validate_authority(&hierarchy)
            .is_ok());
        assert!(envelope("region", "rack-a")
            .validate_authority(&hierarchy)
            .is_ok());
    }

    #[test]
    fn sibling_cannot_constrain_sibling() {
        let hierarchy = hierarchy();
        let result = envelope("rack-a", "rack-b").validate_authority(&hierarchy);
        assert!(matches!(result, Err(EnvelopeError::IssuerNotAncestor { .. })));
    }

    #[test]
    fn valid_operating_point_passes() {
        let hierarchy = hierarchy();
        let result = envelope("site", "rack-a")
            .evaluate(&hierarchy, &point(), t0() + Duration::minutes(10))
            .unwrap();
        assert!(result.compliant);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn expiry_fails_closed() {
        let hierarchy = hierarchy();
        let result = envelope("site", "rack-a")
            .evaluate(&hierarchy, &point(), t0() + Duration::hours(1))
            .unwrap();
        assert!(!result.compliant);
        assert!(result.violations.contains(&EnvelopeViolation::Expired));
    }

    #[test]
    fn missing_observation_fails_closed() {
        let hierarchy = hierarchy();
        let mut point = point();
        point.resources.clear();
        let result = envelope("site", "rack-a")
            .evaluate(&hierarchy, &point, t0() + Duration::minutes(10))
            .unwrap();
        assert!(!result.compliant);
        assert!(matches!(
            result.violations.as_slice(),
            [EnvelopeViolation::MissingObservation { .. }]
        ));
    }

    #[test]
    fn resource_upper_bound_is_enforced() {
        let hierarchy = hierarchy();
        let mut point = point();
        point.resources[0].value = 120.0;
        let result = envelope("site", "rack-a")
            .evaluate(&hierarchy, &point, t0() + Duration::minutes(10))
            .unwrap();
        assert!(result.violations.iter().any(|violation| matches!(
            violation,
            EnvelopeViolation::AboveMaximum { observed, maximum, .. }
                if *observed == 120.0 && *maximum == 100.0
        )));
    }

    #[test]
    fn service_floor_and_shedding_are_independent_hard_bounds() {
        let hierarchy = hierarchy();
        let mut point = point();
        point.critical_service_fraction = 0.7;
        point.shed_fraction = 0.3;
        let result = envelope("site", "rack-a")
            .evaluate(&hierarchy, &point, t0() + Duration::minutes(10))
            .unwrap();
        assert!(result.violations.iter().any(|violation| matches!(
            violation,
            EnvelopeViolation::CriticalServiceBelowFloor { .. }
        )));
        assert!(result.violations.iter().any(|violation| matches!(
            violation,
            EnvelopeViolation::ShedFractionAboveMaximum { .. }
        )));
    }
}
