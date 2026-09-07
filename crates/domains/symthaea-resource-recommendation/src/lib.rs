// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-set recommendation contract over deterministic resource feasibility.
//!
//! A selector may recommend exactly one candidate identity already present in the
//! feasible subset produced by `symthaea-resource-feasible-set`, or explicitly
//! abstain. The selector cannot introduce a new allocation, mutate the feasible
//! universe, or claim that multiple independently feasible alternatives form a
//! jointly feasible bundle.
//!
//! This crate validates recommendation membership only. Selector identity is an
//! opaque correlation label, not authentication or institutional authority.

#![deny(unsafe_code)]

use symthaea_resource_feasibility::FeasibleResourceCandidate;
use symthaea_resource_feasible_set::DeterministicFeasibleSet;
use thiserror::Error;

/// Untrusted planner/selector output.
///
/// This is intentionally not a stable wire format and carries no score, objective,
/// lease, or authority token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourcePlannerProposal {
    Recommend {
        selector_id: String,
        candidate_id: String,
    },
    Abstain {
        selector_id: String,
    },
}

/// Positive non-serializable proof that one selector recommendation names an
/// exact candidate inside one exact retained deterministic feasible set.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedResourceRecommendation {
    selector_id: String,
    candidate_id: String,
    feasible_set: DeterministicFeasibleSet,
}

impl ValidatedResourceRecommendation {
    pub fn selector_id(&self) -> &str {
        &self.selector_id
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn feasible_set(&self) -> &DeterministicFeasibleSet {
        &self.feasible_set
    }

    pub fn selected_candidate(&self) -> &FeasibleResourceCandidate {
        self.feasible_set
            .feasible_candidate(&self.candidate_id)
            .expect("validated recommendation must retain selected feasible candidate")
    }
}

/// Validated explicit abstention. Abstention is permitted by the generic contract
/// even when feasible alternatives exist; whether a particular selector is
/// required to choose is a later policy question.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedResourceAbstention {
    selector_id: String,
    feasible_set: DeterministicFeasibleSet,
}

impl ValidatedResourceAbstention {
    pub fn selector_id(&self) -> &str {
        &self.selector_id
    }

    pub fn feasible_set(&self) -> &DeterministicFeasibleSet {
        &self.feasible_set
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidatedResourcePlannerDecision {
    Recommend(ValidatedResourceRecommendation),
    Abstain(ValidatedResourceAbstention),
}

/// Validate one selector output against a closed feasible set.
///
/// The validated recommendation retains the complete feasible set by value. It
/// therefore cannot be detached from the exact base load, discovery universe, and
/// feasibility results against which membership was established.
pub fn validate_resource_planner_proposal(
    feasible_set: &DeterministicFeasibleSet,
    proposal: ResourcePlannerProposal,
) -> Result<ValidatedResourcePlannerDecision, RecommendationError> {
    match proposal {
        ResourcePlannerProposal::Recommend {
            selector_id,
            candidate_id,
        } => {
            validate_selector_id(&selector_id)?;
            if candidate_id.trim().is_empty() {
                return Err(RecommendationError::EmptyCandidateId);
            }

            if feasible_set.feasible_candidate(&candidate_id).is_some() {
                return Ok(ValidatedResourcePlannerDecision::Recommend(
                    ValidatedResourceRecommendation {
                        selector_id,
                        candidate_id,
                        feasible_set: feasible_set.clone(),
                    },
                ));
            }

            if feasible_set.rejection(&candidate_id).is_some() {
                return Err(RecommendationError::RejectedCandidate(candidate_id));
            }

            Err(RecommendationError::CandidateOutsideUniverse(candidate_id))
        }
        ResourcePlannerProposal::Abstain { selector_id } => {
            validate_selector_id(&selector_id)?;
            Ok(ValidatedResourcePlannerDecision::Abstain(
                ValidatedResourceAbstention {
                    selector_id,
                    feasible_set: feasible_set.clone(),
                },
            ))
        }
    }
}

fn validate_selector_id(selector_id: &str) -> Result<(), RecommendationError> {
    if selector_id.trim().is_empty() {
        Err(RecommendationError::EmptySelectorId)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RecommendationError {
    #[error("selector id must not be empty")]
    EmptySelectorId,
    #[error("candidate id must not be empty")]
    EmptyCandidateId,
    #[error("candidate {0} exists in the universe but was rejected by feasibility")]
    RejectedCandidate(String),
    #[error("candidate {0} is outside the retained candidate universe")]
    CandidateOutsideUniverse(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, TimeZone, Utc};
    use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_feasible_set::enumerate_feasible_set;
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
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

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn topology() -> ResourceTopology {
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
        topology
    }

    fn capacities(topology: &ResourceTopology) -> CapacitySchedule {
        let mut schedule = CapacitySchedule::default();
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
            schedule
                .add_window(
                    topology,
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
        schedule
    }

    fn evidence(topology: &ResourceTopology) -> QualityEvidenceSet {
        let mut profile = ResourceQualityProfile::new(power(1.0).key);
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, 40.0)
            .unwrap();
        let mut evidence = QualityEvidenceSet::default();
        evidence
            .insert(
                topology,
                QualityEvidenceWindow {
                    id: "quality".into(),
                    subject: QualitySubject::Link {
                        link_id: "line".into(),
                    },
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    profile,
                    evidence_class: QualityEvidenceClass::Observed,
                    evidence_ref: "sensor:quality".into(),
                },
            )
            .unwrap();
        evidence
    }

    fn requirement() -> ResourceQualityRequirement {
        let mut requirement = ResourceQualityRequirement::new(power(1.0).key);
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(30.0),
                maximum: Some(60.0),
            })
            .unwrap();
        requirement
    }

    fn policy() -> QualityQualificationPolicy {
        QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        }
    }

    fn allocation(id: &str, value: f64) -> PlannedAllocation {
        PlannedAllocation {
            id: id.into(),
            link_id: "line".into(),
            valid_from: t0(),
            valid_until: t0() + Duration::hours(1),
            sent: power(value),
        }
    }

    fn set() -> DeterministicFeasibleSet {
        let topology = topology();
        enumerate_feasible_set(
            &topology,
            &capacities(&topology),
            &AllocationBook::default(),
            &evidence(&topology),
            &requirement(),
            &policy(),
            vec![allocation("feasible-a", 40.0), allocation("rejected", 120.0)],
            8,
        )
        .unwrap()
    }

    #[test]
    fn recommendation_can_only_name_existing_feasible_candidate() {
        let set = set();
        let decision = validate_resource_planner_proposal(
            &set,
            ResourcePlannerProposal::Recommend {
                selector_id: "baseline".into(),
                candidate_id: "feasible-a".into(),
            },
        )
        .unwrap();

        let ValidatedResourcePlannerDecision::Recommend(validated) = decision else {
            panic!("expected validated recommendation");
        };
        assert_eq!(validated.selector_id(), "baseline");
        assert_eq!(validated.candidate_id(), "feasible-a");
        assert_eq!(validated.selected_candidate().candidate_id(), "feasible-a");
        assert_eq!(validated.feasible_set(), &set);
    }

    #[test]
    fn rejected_candidate_cannot_be_recommended() {
        let result = validate_resource_planner_proposal(
            &set(),
            ResourcePlannerProposal::Recommend {
                selector_id: "pareto".into(),
                candidate_id: "rejected".into(),
            },
        );
        assert_eq!(
            result,
            Err(RecommendationError::RejectedCandidate("rejected".into()))
        );
    }

    #[test]
    fn selector_cannot_manufacture_candidate_outside_universe() {
        let result = validate_resource_planner_proposal(
            &set(),
            ResourcePlannerProposal::Recommend {
                selector_id: "hdc-shadow".into(),
                candidate_id: "invented".into(),
            },
        );
        assert_eq!(
            result,
            Err(RecommendationError::CandidateOutsideUniverse("invented".into()))
        );
    }

    #[test]
    fn empty_selector_identity_fails_closed() {
        let result = validate_resource_planner_proposal(
            &set(),
            ResourcePlannerProposal::Recommend {
                selector_id: "  ".into(),
                candidate_id: "feasible-a".into(),
            },
        );
        assert_eq!(result, Err(RecommendationError::EmptySelectorId));
    }

    #[test]
    fn empty_candidate_identity_fails_closed() {
        let result = validate_resource_planner_proposal(
            &set(),
            ResourcePlannerProposal::Recommend {
                selector_id: "baseline".into(),
                candidate_id: "".into(),
            },
        );
        assert_eq!(result, Err(RecommendationError::EmptyCandidateId));
    }

    #[test]
    fn abstention_is_explicit_and_retains_exact_feasible_set() {
        let set = set();
        let decision = validate_resource_planner_proposal(
            &set,
            ResourcePlannerProposal::Abstain {
                selector_id: "experimental".into(),
            },
        )
        .unwrap();
        let ValidatedResourcePlannerDecision::Abstain(abstention) = decision else {
            panic!("expected abstention");
        };
        assert_eq!(abstention.selector_id(), "experimental");
        assert_eq!(abstention.feasible_set(), &set);
    }

    #[test]
    fn contract_has_no_multi_candidate_recommendation_path() {
        let set = set();
        let decision = validate_resource_planner_proposal(
            &set,
            ResourcePlannerProposal::Recommend {
                selector_id: "single-choice".into(),
                candidate_id: "feasible-a".into(),
            },
        )
        .unwrap();
        let ValidatedResourcePlannerDecision::Recommend(validated) = decision else {
            panic!("expected recommendation");
        };
        // The positive witness resolves one exact candidate only. Any future
        // multi-candidate selection requires a separate bundle-feasibility theorem.
        assert_eq!(validated.selected_candidate().candidate_id(), "feasible-a");
    }
}
