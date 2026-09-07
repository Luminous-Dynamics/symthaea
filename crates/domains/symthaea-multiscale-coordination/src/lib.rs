// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded Pareto coordination across direct-child infrastructure nodes.
//!
//! A coordinator ranks whole parent-level plans. Each admissible plan contains
//! exactly one expiring `OperatingEnvelope` for every direct child of the issuer.
//! It never emits actuator commands and never reaches through a child to control a
//! deeper descendant. This preserves locality while still allowing larger scales to
//! coordinate trade-offs among bounded alternatives.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use symthaea_operating_envelope::OperatingEnvelope;
use symthaea_operations_research::{
    ParetoCandidate, ParetoError, ParetoObjective, rank_pareto,
};
use symthaea_resource_hierarchy::ResourceHierarchy;
use thiserror::Error;

/// One parent-level alternative containing a complete set of child envelopes.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinationPlan {
    pub id: String,
    pub objective_values: Vec<f64>,
    pub child_envelopes: Vec<OperatingEnvelope>,
}

impl CoordinationPlan {
    pub fn new(
        id: impl Into<String>,
        objective_values: Vec<f64>,
        child_envelopes: Vec<OperatingEnvelope>,
    ) -> Self {
        Self {
            id: id.into(),
            objective_values,
            child_envelopes,
        }
    }
}

/// Pareto-ranked coordination alternatives. Front zero is non-dominated.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinationRanking {
    pub issuer_node_id: String,
    pub fronts: Vec<Vec<CoordinationPlan>>,
}

impl CoordinationRanking {
    pub fn frontier(&self) -> &[CoordinationPlan] {
        self.fronts.first().map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Validate and Pareto-rank complete direct-child coordination plans.
pub fn rank_coordination_plans(
    hierarchy: &ResourceHierarchy,
    issuer_node_id: &str,
    objectives: &[ParetoObjective],
    plans: &[CoordinationPlan],
) -> Result<CoordinationRanking, CoordinationError> {
    if hierarchy.node(issuer_node_id).is_none() {
        return Err(CoordinationError::UnknownIssuer(issuer_node_id.to_owned()));
    }

    let direct_children: BTreeSet<String> = hierarchy
        .children(issuer_node_id)
        .map(str::to_owned)
        .collect();
    if direct_children.is_empty() {
        return Err(CoordinationError::NoDirectChildren(
            issuer_node_id.to_owned(),
        ));
    }

    let mut plans_by_id = BTreeMap::new();
    for plan in plans {
        validate_plan(hierarchy, issuer_node_id, &direct_children, plan)?;
        if plans_by_id.insert(plan.id.clone(), plan.clone()).is_some() {
            return Err(CoordinationError::DuplicatePlan(plan.id.clone()));
        }
    }

    let pareto_candidates: Vec<ParetoCandidate> = plans
        .iter()
        .map(|plan| ParetoCandidate::new(plan.id.clone(), plan.objective_values.clone()))
        .collect();
    let ranking = rank_pareto(objectives, &pareto_candidates)?;

    let fronts = ranking
        .fronts
        .into_iter()
        .map(|front| {
            front
                .into_iter()
                .map(|candidate| {
                    plans_by_id
                        .get(&candidate.id)
                        .expect("validated Pareto candidate must map to a plan")
                        .clone()
                })
                .collect()
        })
        .collect();

    Ok(CoordinationRanking {
        issuer_node_id: issuer_node_id.to_owned(),
        fronts,
    })
}

fn validate_plan(
    hierarchy: &ResourceHierarchy,
    issuer_node_id: &str,
    direct_children: &BTreeSet<String>,
    plan: &CoordinationPlan,
) -> Result<(), CoordinationError> {
    if plan.id.is_empty() {
        return Err(CoordinationError::EmptyPlanId);
    }

    let mut subjects = BTreeSet::new();
    let mut generation = None;
    let mut latest_start = None;
    let mut earliest_end = None;

    for envelope in &plan.child_envelopes {
        if envelope.issuer_node_id != issuer_node_id {
            return Err(CoordinationError::WrongIssuer {
                plan_id: plan.id.clone(),
                envelope_id: envelope.id.clone(),
                expected: issuer_node_id.to_owned(),
                actual: envelope.issuer_node_id.clone(),
            });
        }
        if !direct_children.contains(&envelope.subject_node_id) {
            return Err(CoordinationError::NotDirectChild {
                plan_id: plan.id.clone(),
                subject: envelope.subject_node_id.clone(),
            });
        }
        if !subjects.insert(envelope.subject_node_id.clone()) {
            return Err(CoordinationError::DuplicateChildEnvelope {
                plan_id: plan.id.clone(),
                subject: envelope.subject_node_id.clone(),
            });
        }

        envelope
            .validate_structure()
            .map_err(|error| CoordinationError::InvalidEnvelope {
                plan_id: plan.id.clone(),
                envelope_id: envelope.id.clone(),
                reason: error.to_string(),
            })?;
        envelope
            .validate_authority(hierarchy)
            .map_err(|error| CoordinationError::InvalidEnvelope {
                plan_id: plan.id.clone(),
                envelope_id: envelope.id.clone(),
                reason: error.to_string(),
            })?;

        match generation {
            None => generation = Some(envelope.generation),
            Some(expected) if expected != envelope.generation => {
                return Err(CoordinationError::MixedGeneration {
                    plan_id: plan.id.clone(),
                    expected,
                    actual: envelope.generation,
                });
            }
            Some(_) => {}
        }

        latest_start = Some(match latest_start {
            None => envelope.valid_from,
            Some(current) => current.max(envelope.valid_from),
        });
        earliest_end = Some(match earliest_end {
            None => envelope.valid_until,
            Some(current) => current.min(envelope.valid_until),
        });
    }

    for child in direct_children {
        if !subjects.contains(child) {
            return Err(CoordinationError::MissingChildEnvelope {
                plan_id: plan.id.clone(),
                child: child.clone(),
            });
        }
    }

    if let (Some(start), Some(end)) = (latest_start, earliest_end)
        && end <= start
    {
        return Err(CoordinationError::NoCommonValidityWindow(plan.id.clone()));
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CoordinationError {
    #[error("unknown coordination issuer {0}")]
    UnknownIssuer(String),
    #[error("coordination issuer {0} has no direct children")]
    NoDirectChildren(String),
    #[error("coordination plan id must not be empty")]
    EmptyPlanId,
    #[error("duplicate coordination plan id {0}")]
    DuplicatePlan(String),
    #[error("plan {plan_id} envelope {envelope_id} issuer {actual} does not match {expected}")]
    WrongIssuer {
        plan_id: String,
        envelope_id: String,
        expected: String,
        actual: String,
    },
    #[error("plan {plan_id} attempts to coordinate non-direct child {subject}")]
    NotDirectChild { plan_id: String, subject: String },
    #[error("plan {plan_id} contains more than one envelope for child {subject}")]
    DuplicateChildEnvelope { plan_id: String, subject: String },
    #[error("plan {plan_id} is missing an envelope for direct child {child}")]
    MissingChildEnvelope { plan_id: String, child: String },
    #[error("plan {plan_id} envelope {envelope_id} is invalid: {reason}")]
    InvalidEnvelope {
        plan_id: String,
        envelope_id: String,
        reason: String,
    },
    #[error("plan {plan_id} mixes policy generations {expected} and {actual}")]
    MixedGeneration {
        plan_id: String,
        expected: u64,
        actual: u64,
    },
    #[error("plan {0} has no time interval in which all child envelopes are valid")]
    NoCommonValidityWindow(String),
    #[error(transparent)]
    Pareto(#[from] ParetoError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use std::collections::BTreeSet;
    use symthaea_operating_envelope::{OperatingMode, ResourceConstraint};
    use symthaea_operations_research::ObjectiveDirection;
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::ResourceEnvelope;

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
                    "site-a",
                    "Site A",
                    NodeScale::Site,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "region",
                ResourceNode::new(
                    "site-b",
                    "Site B",
                    NodeScale::Site,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
            .insert_child(
                "site-a",
                ResourceNode::new(
                    "rack-a",
                    "Rack A",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn envelope(issuer: &str, subject: &str, generation: u64, offset_minutes: i64) -> OperatingEnvelope {
        let start = Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
            + Duration::minutes(offset_minutes);
        let mut modes = BTreeSet::new();
        modes.insert(OperatingMode::Normal);
        OperatingEnvelope {
            id: format!("{issuer}-{subject}-{generation}-{offset_minutes}"),
            issuer_node_id: issuer.into(),
            subject_node_id: subject.into(),
            generation,
            valid_from: start,
            valid_until: start + Duration::hours(1),
            allowed_modes: modes,
            resource_constraints: Vec::<ResourceConstraint>::new(),
            critical_service_floor: 0.8,
            max_shed_fraction: 0.2,
        }
    }

    fn objectives() -> Vec<ParetoObjective> {
        vec![
            ParetoObjective::new("cost", ObjectiveDirection::Minimize),
            ParetoObjective::new("resilience", ObjectiveDirection::Maximize),
        ]
    }

    fn plan(id: &str, values: Vec<f64>) -> CoordinationPlan {
        CoordinationPlan::new(
            id,
            values,
            vec![
                envelope("region", "site-a", 7, 0),
                envelope("region", "site-b", 7, 0),
            ],
        )
    }

    #[test]
    fn ranks_complete_parent_level_tradeoffs() {
        let hierarchy = hierarchy();
        let plans = vec![
            plan("cheap", vec![1.0, 4.0]),
            plan("resilient", vec![4.0, 9.0]),
            plan("dominated", vec![5.0, 3.0]),
        ];
        let ranking = rank_coordination_plans(&hierarchy, "region", &objectives(), &plans).unwrap();
        let ids: Vec<&str> = ranking
            .frontier()
            .iter()
            .map(|plan| plan.id.as_str())
            .collect();
        assert_eq!(ids, vec!["cheap", "resilient"]);
    }

    #[test]
    fn parent_cannot_reach_through_child_to_grandchild() {
        let hierarchy = hierarchy();
        let bad = CoordinationPlan::new(
            "bad",
            vec![1.0, 1.0],
            vec![
                envelope("region", "rack-a", 7, 0),
                envelope("region", "site-b", 7, 0),
            ],
        );
        assert!(matches!(
            rank_coordination_plans(&hierarchy, "region", &objectives(), &[bad]),
            Err(CoordinationError::NotDirectChild { subject, .. }) if subject == "rack-a"
        ));
    }

    #[test]
    fn every_direct_child_must_receive_one_envelope() {
        let hierarchy = hierarchy();
        let incomplete = CoordinationPlan::new(
            "incomplete",
            vec![1.0, 1.0],
            vec![envelope("region", "site-a", 7, 0)],
        );
        assert!(matches!(
            rank_coordination_plans(&hierarchy, "region", &objectives(), &[incomplete]),
            Err(CoordinationError::MissingChildEnvelope { child, .. }) if child == "site-b"
        ));
    }

    #[test]
    fn duplicate_child_authority_is_rejected() {
        let hierarchy = hierarchy();
        let duplicate = CoordinationPlan::new(
            "duplicate",
            vec![1.0, 1.0],
            vec![
                envelope("region", "site-a", 7, 0),
                envelope("region", "site-a", 7, 0),
                envelope("region", "site-b", 7, 0),
            ],
        );
        assert!(matches!(
            rank_coordination_plans(&hierarchy, "region", &objectives(), &[duplicate]),
            Err(CoordinationError::DuplicateChildEnvelope { .. })
        ));
    }

    #[test]
    fn mixed_policy_generation_is_rejected() {
        let hierarchy = hierarchy();
        let mixed = CoordinationPlan::new(
            "mixed",
            vec![1.0, 1.0],
            vec![
                envelope("region", "site-a", 7, 0),
                envelope("region", "site-b", 8, 0),
            ],
        );
        assert!(matches!(
            rank_coordination_plans(&hierarchy, "region", &objectives(), &[mixed]),
            Err(CoordinationError::MixedGeneration { .. })
        ));
    }

    #[test]
    fn child_envelopes_need_a_shared_validity_interval() {
        let hierarchy = hierarchy();
        let disjoint = CoordinationPlan::new(
            "disjoint",
            vec![1.0, 1.0],
            vec![
                envelope("region", "site-a", 7, 0),
                envelope("region", "site-b", 7, 60),
            ],
        );
        assert!(matches!(
            rank_coordination_plans(&hierarchy, "region", &objectives(), &[disjoint]),
            Err(CoordinationError::NoCommonValidityWindow(id)) if id == "disjoint"
        ));
    }

    #[test]
    fn leaf_node_cannot_act_as_a_coordinator() {
        let hierarchy = hierarchy();
        assert!(matches!(
            rank_coordination_plans(&hierarchy, "rack-a", &objectives(), &[]),
            Err(CoordinationError::NoDirectChildren(id)) if id == "rack-a"
        ));
    }
}
