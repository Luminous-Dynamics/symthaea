// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Locally-owned survival envelopes composed with expiring parent coordination.
//!
//! A higher-scale coordinator may tighten a child's runtime operating envelope, but
//! loss of that coordinator must not erase the child's locally commissioned safety
//! contract. Effective operation therefore satisfies the local survival envelope
//! unconditionally and, when present, the current admitted parent lease as well.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_operating_authority::OperatingAuthorityRegistry;
use symthaea_operating_envelope::{
    EnvelopeViolation, FractionMetric, OperatingMode, OperatingPoint, ResourceConstraint,
};
use symthaea_resource_hierarchy::ResourceHierarchy;
use thiserror::Error;

/// Non-expiring local safety contract provisioned with the node itself.
///
/// This is intentionally not an `OperatingEnvelope`: it has no parent issuer,
/// generation, or remote validity window. Parent coordination is evaluated in
/// addition to this contract and can never relax it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalSafetyEnvelope {
    pub id: String,
    pub subject_node_id: String,
    pub allowed_modes: BTreeSet<OperatingMode>,
    pub resource_constraints: Vec<ResourceConstraint>,
    pub critical_service_floor: f64,
    pub max_shed_fraction: f64,
}

impl LocalSafetyEnvelope {
    pub fn validate(&self, hierarchy: &ResourceHierarchy) -> Result<(), AutonomyError> {
        if self.id.trim().is_empty() {
            return Err(AutonomyError::EmptyLocalEnvelopeId);
        }
        if hierarchy.node(&self.subject_node_id).is_none() {
            return Err(AutonomyError::UnknownSubject(self.subject_node_id.clone()));
        }
        if self.allowed_modes.is_empty() {
            return Err(AutonomyError::NoAllowedModes);
        }
        validate_fraction("critical_service_floor", self.critical_service_floor)?;
        validate_fraction("max_shed_fraction", self.max_shed_fraction)?;
        for constraint in &self.resource_constraints {
            constraint
                .validate()
                .map_err(|error| AutonomyError::InvalidLocalConstraint(error.to_string()))?;
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        hierarchy: &ResourceHierarchy,
        point: &OperatingPoint,
    ) -> Result<Vec<EnvelopeViolation>, AutonomyError> {
        self.validate(hierarchy)?;
        let mut violations = Vec::new();

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
            let observation = point.resources.iter().find(|observation| {
                observation.key == constraint.key && observation.metric == constraint.metric
            });
            let Some(observation) = observation else {
                violations.push(EnvelopeViolation::MissingObservation {
                    key: constraint.key,
                    metric: constraint.metric,
                });
                continue;
            };
            let observed = observation.value;
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

        Ok(violations)
    }
}

/// Whether the node is operating from its own survival contract alone or with an
/// additional currently-admitted parent coordination lease.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationMode {
    LocalOnly,
    ParentCoordinated {
        issuer_node_id: String,
        envelope_id: String,
        generation: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveOperatingEvaluation {
    pub compliant: bool,
    pub mode: CoordinationMode,
    pub local_violations: Vec<EnvelopeViolation>,
    pub parent_violations: Vec<EnvelopeViolation>,
}

/// Evaluate a point against the non-expiring local survival contract and, when one
/// is currently active, the admitted parent lease.
///
/// A missing/expired parent lease does not erase local authority. Conversely, a
/// parent lease can never relax a local limit because both sets of constraints must
/// be satisfied when parent coordination is active.
pub fn evaluate_effective_operation(
    local: &LocalSafetyEnvelope,
    hierarchy: &ResourceHierarchy,
    authority: &OperatingAuthorityRegistry,
    point: &OperatingPoint,
    now: DateTime<Utc>,
) -> Result<EffectiveOperatingEvaluation, AutonomyError> {
    let local_violations = local.evaluate(hierarchy, point)?;

    if let Some(parent) = authority.active(hierarchy, &local.subject_node_id, now) {
        let parent_evaluation = parent
            .evaluate(hierarchy, point, now)
            .map_err(|error| AutonomyError::ParentEvaluation(error.to_string()))?;
        let compliant = local_violations.is_empty() && parent_evaluation.compliant;
        Ok(EffectiveOperatingEvaluation {
            compliant,
            mode: CoordinationMode::ParentCoordinated {
                issuer_node_id: parent.issuer_node_id.clone(),
                envelope_id: parent.id.clone(),
                generation: parent.generation,
            },
            local_violations,
            parent_violations: parent_evaluation.violations,
        })
    } else {
        Ok(EffectiveOperatingEvaluation {
            compliant: local_violations.is_empty(),
            mode: CoordinationMode::LocalOnly,
            local_violations,
            parent_violations: Vec::new(),
        })
    }
}

fn validate_fraction(label: &'static str, value: f64) -> Result<(), AutonomyError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(AutonomyError::InvalidFraction { label, value });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AutonomyError {
    #[error("local safety envelope id must not be empty")]
    EmptyLocalEnvelopeId,
    #[error("unknown local safety subject {0}")]
    UnknownSubject(String),
    #[error("local safety envelope must allow at least one operating mode")]
    NoAllowedModes,
    #[error("invalid local fraction {label}: {value}")]
    InvalidFraction { label: &'static str, value: f64 },
    #[error("invalid local resource constraint: {0}")]
    InvalidLocalConstraint(String),
    #[error("parent operating-envelope evaluation failed: {0}")]
    ParentEvaluation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_operating_authority::LeaseAdmission;
    use symthaea_operating_envelope::{
        BoundaryMetric, ObservedResourceMetric, OperatingEnvelope, ResourceConstraint,
    };
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::{ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit};

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power_key() -> symthaea_resource_model::ResourceKey {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack",
                    "Rack",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn local() -> LocalSafetyEnvelope {
        let mut allowed_modes = BTreeSet::new();
        allowed_modes.insert(OperatingMode::Normal);
        allowed_modes.insert(OperatingMode::Islanded);
        allowed_modes.insert(OperatingMode::Emergency);
        LocalSafetyEnvelope {
            id: "rack-survival-v1".into(),
            subject_node_id: "rack".into(),
            allowed_modes,
            resource_constraints: vec![ResourceConstraint {
                key: power_key(),
                metric: BoundaryMetric::Import,
                minimum: None,
                maximum: Some(100.0),
            }],
            critical_service_floor: 0.70,
            max_shed_fraction: 0.30,
        }
    }

    fn point(import: f64) -> OperatingPoint {
        OperatingPoint {
            mode: OperatingMode::Normal,
            critical_service_fraction: 0.85,
            shed_fraction: 0.10,
            resources: vec![ObservedResourceMetric {
                key: power_key(),
                metric: BoundaryMetric::Import,
                value: import,
            }],
        }
    }

    fn parent(max_import: f64) -> OperatingEnvelope {
        let mut allowed_modes = BTreeSet::new();
        allowed_modes.insert(OperatingMode::Normal);
        OperatingEnvelope {
            id: format!("parent-{max_import}"),
            issuer_node_id: "site".into(),
            subject_node_id: "rack".into(),
            generation: 1,
            valid_from: t0(),
            valid_until: t0() + Duration::hours(1),
            allowed_modes,
            resource_constraints: vec![ResourceConstraint {
                key: power_key(),
                metric: BoundaryMetric::Import,
                minimum: None,
                maximum: Some(max_import),
            }],
            critical_service_floor: 0.70,
            max_shed_fraction: 0.30,
        }
    }

    #[test]
    fn local_contract_keeps_node_operable_without_parent_lease() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let evaluation = evaluate_effective_operation(
            &local(),
            &hierarchy,
            &authority,
            &point(90.0),
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert!(evaluation.compliant);
        assert_eq!(evaluation.mode, CoordinationMode::LocalOnly);
    }

    #[test]
    fn parent_can_tighten_local_operation() {
        let hierarchy = hierarchy();
        let mut authority = OperatingAuthorityRegistry::new();
        assert_eq!(
            authority
                .admit(
                    &hierarchy,
                    parent(80.0),
                    t0() + Duration::minutes(1),
                )
                .unwrap(),
            LeaseAdmission::Accepted
        );
        let evaluation = evaluate_effective_operation(
            &local(),
            &hierarchy,
            &authority,
            &point(90.0),
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert!(!evaluation.compliant);
        assert!(evaluation.local_violations.is_empty());
        assert!(evaluation.parent_violations.iter().any(|violation| matches!(
            violation,
            EnvelopeViolation::AboveMaximum { maximum, .. } if *maximum == 80.0
        )));
    }

    #[test]
    fn parent_cannot_relax_local_safety_limit() {
        let hierarchy = hierarchy();
        let mut authority = OperatingAuthorityRegistry::new();
        authority
            .admit(
                &hierarchy,
                parent(120.0),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        let evaluation = evaluate_effective_operation(
            &local(),
            &hierarchy,
            &authority,
            &point(110.0),
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert!(!evaluation.compliant);
        assert!(evaluation.parent_violations.is_empty());
        assert!(evaluation.local_violations.iter().any(|violation| matches!(
            violation,
            EnvelopeViolation::AboveMaximum { maximum, .. } if *maximum == 100.0
        )));
    }

    #[test]
    fn expired_parent_coordination_falls_back_to_local_contract() {
        let hierarchy = hierarchy();
        let mut authority = OperatingAuthorityRegistry::new();
        authority
            .admit(
                &hierarchy,
                parent(80.0),
                t0() + Duration::minutes(1),
            )
            .unwrap();
        let evaluation = evaluate_effective_operation(
            &local(),
            &hierarchy,
            &authority,
            &point(90.0),
            t0() + Duration::hours(2),
        )
        .unwrap();
        assert!(evaluation.compliant);
        assert_eq!(evaluation.mode, CoordinationMode::LocalOnly);
    }

    #[test]
    fn local_violation_still_blocks_when_parent_is_absent() {
        let hierarchy = hierarchy();
        let authority = OperatingAuthorityRegistry::new();
        let evaluation = evaluate_effective_operation(
            &local(),
            &hierarchy,
            &authority,
            &point(110.0),
            t0() + Duration::hours(2),
        )
        .unwrap();
        assert!(!evaluation.compliant);
        assert!(evaluation.parent_violations.is_empty());
    }
}
