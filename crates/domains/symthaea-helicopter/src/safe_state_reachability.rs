// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded abstract safe-state reachability analysis.
//!
//! Runtime monitors check observed deadlines; this module checks a different
//! property before deployment: for each declared fault case, does the abstract
//! contingency graph contain a capability-feasible path to a declared safe
//! terminal before the case deadline? This is finite-state assurance, not a
//! proof of continuous helicopter dynamics.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AbstractFlightState {
    NominalMission,
    DegradedHover,
    ReturnToBase,
    LandingApproach,
    Autorotation,
    GroundSafe,
    GroundUnsafe,
    Uncontrolled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SafetyCapability {
    Navigation,
    MainRotorPower,
    TailRotorAuthority,
    RotorEnergy,
    LandingZoneKnown,
    ControlComputer,
    ActuatorPower,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeStateTransition {
    pub from: AbstractFlightState,
    pub to: AbstractFlightState,
    pub maximum_duration_s: f64,
    pub required_capabilities: Vec<SafetyCapability>,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReachabilityCase {
    pub case_id: String,
    pub initial_state: AbstractFlightState,
    pub available_capabilities: Vec<SafetyCapability>,
    pub deadline_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReachabilityStatus {
    Reachable,
    Unreachable,
    DeadlineExceeded,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeStateReachabilityReport {
    pub case_id: String,
    pub status: ReachabilityStatus,
    pub reached_safe_state: Option<AbstractFlightState>,
    pub worst_case_duration_s: Option<f64>,
    pub path: Vec<AbstractFlightState>,
    pub missing_capabilities: Vec<SafetyCapability>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeStateReachabilityError {
    EmptySafeStateSet,
    InvalidTransition,
    InvalidCase,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeStateReachabilityModel {
    pub transitions: Vec<SafeStateTransition>,
    pub safe_states: Vec<AbstractFlightState>,
}

impl SafeStateReachabilityModel {
    pub fn new(
        transitions: Vec<SafeStateTransition>,
        safe_states: Vec<AbstractFlightState>,
    ) -> Result<Self, SafeStateReachabilityError> {
        let model = Self {
            transitions,
            safe_states,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn default_helicopter_model() -> Self {
        Self::new(
            vec![
                transition(
                    AbstractFlightState::NominalMission,
                    AbstractFlightState::DegradedHover,
                    2.0,
                    &[
                        SafetyCapability::ControlComputer,
                        SafetyCapability::ActuatorPower,
                    ],
                    "stabilize before choosing a contingency",
                ),
                transition(
                    AbstractFlightState::NominalMission,
                    AbstractFlightState::ReturnToBase,
                    1.0,
                    &[
                        SafetyCapability::Navigation,
                        SafetyCapability::MainRotorPower,
                        SafetyCapability::ControlComputer,
                    ],
                    "retain powered navigation to the protected base",
                ),
                transition(
                    AbstractFlightState::DegradedHover,
                    AbstractFlightState::LandingApproach,
                    15.0,
                    &[
                        SafetyCapability::Navigation,
                        SafetyCapability::MainRotorPower,
                        SafetyCapability::LandingZoneKnown,
                    ],
                    "enter a powered landing approach",
                ),
                transition(
                    AbstractFlightState::DegradedHover,
                    AbstractFlightState::Autorotation,
                    1.0,
                    &[
                        SafetyCapability::RotorEnergy,
                        SafetyCapability::ControlComputer,
                    ],
                    "lower collective and preserve rotor energy after power loss",
                ),
                transition(
                    AbstractFlightState::ReturnToBase,
                    AbstractFlightState::LandingApproach,
                    300.0,
                    &[
                        SafetyCapability::Navigation,
                        SafetyCapability::MainRotorPower,
                        SafetyCapability::LandingZoneKnown,
                    ],
                    "navigate from the operating area to the base approach",
                ),
                transition(
                    AbstractFlightState::LandingApproach,
                    AbstractFlightState::GroundSafe,
                    60.0,
                    &[
                        SafetyCapability::LandingZoneKnown,
                        SafetyCapability::ActuatorPower,
                    ],
                    "complete a controlled touchdown and disarm",
                ),
                transition(
                    AbstractFlightState::Autorotation,
                    AbstractFlightState::GroundSafe,
                    90.0,
                    &[
                        SafetyCapability::RotorEnergy,
                        SafetyCapability::LandingZoneKnown,
                        SafetyCapability::ActuatorPower,
                    ],
                    "manage rotor energy through flare and touchdown",
                ),
            ],
            vec![AbstractFlightState::GroundSafe],
        )
        .expect("default safe-state model must remain valid")
    }

    pub fn validate(&self) -> Result<(), SafeStateReachabilityError> {
        if self.safe_states.is_empty() {
            return Err(SafeStateReachabilityError::EmptySafeStateSet);
        }
        let safe: BTreeSet<_> = self.safe_states.iter().copied().collect();
        if safe.len() != self.safe_states.len() {
            return Err(SafeStateReachabilityError::EmptySafeStateSet);
        }
        for edge in &self.transitions {
            if !edge.maximum_duration_s.is_finite()
                || edge.maximum_duration_s <= 0.0
                || edge.from == edge.to
                || edge.rationale.trim().is_empty()
            {
                return Err(SafeStateReachabilityError::InvalidTransition);
            }
            let unique: BTreeSet<_> = edge.required_capabilities.iter().copied().collect();
            if unique.len() != edge.required_capabilities.len() {
                return Err(SafeStateReachabilityError::InvalidTransition);
            }
        }
        Ok(())
    }

    pub fn assess(
        &self,
        case: &ReachabilityCase,
    ) -> Result<SafeStateReachabilityReport, SafeStateReachabilityError> {
        self.validate()?;
        if case.case_id.trim().is_empty() || !case.deadline_s.is_finite() || case.deadline_s <= 0.0
        {
            return Err(SafeStateReachabilityError::InvalidCase);
        }
        let capabilities: BTreeSet<_> = case.available_capabilities.iter().copied().collect();
        if capabilities.len() != case.available_capabilities.len() {
            return Err(SafeStateReachabilityError::InvalidCase);
        }
        let safe: BTreeSet<_> = self.safe_states.iter().copied().collect();
        if safe.contains(&case.initial_state) {
            return Ok(SafeStateReachabilityReport {
                case_id: case.case_id.clone(),
                status: ReachabilityStatus::Reachable,
                reached_safe_state: Some(case.initial_state),
                worst_case_duration_s: Some(0.0),
                path: vec![case.initial_state],
                missing_capabilities: Vec::new(),
            });
        }

        let mut distance = BTreeMap::new();
        let mut predecessor = BTreeMap::new();
        let mut frontier = BTreeSet::new();
        distance.insert(case.initial_state, 0.0);
        frontier.insert(case.initial_state);

        while !frontier.is_empty() {
            let current = *frontier
                .iter()
                .min_by(|left, right| {
                    distance
                        .get(*left)
                        .expect("frontier state has a distance")
                        .partial_cmp(distance.get(*right).expect("frontier state has a distance"))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("non-empty frontier has a minimum");
            frontier.remove(&current);
            let current_distance = distance[&current];
            for edge in self.transitions.iter().filter(|edge| edge.from == current) {
                if !edge
                    .required_capabilities
                    .iter()
                    .all(|capability| capabilities.contains(capability))
                {
                    continue;
                }
                let candidate = current_distance + edge.maximum_duration_s;
                if candidate < *distance.get(&edge.to).unwrap_or(&f64::INFINITY) {
                    distance.insert(edge.to, candidate);
                    predecessor.insert(edge.to, current);
                    frontier.insert(edge.to);
                }
            }
        }

        let reached = self
            .safe_states
            .iter()
            .filter_map(|state| distance.get(state).map(|duration| (*state, *duration)))
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((safe_state, duration)) = reached {
            let mut path = vec![safe_state];
            let mut cursor = safe_state;
            while let Some(previous) = predecessor.get(&cursor).copied() {
                path.push(previous);
                cursor = previous;
            }
            path.reverse();
            return Ok(SafeStateReachabilityReport {
                case_id: case.case_id.clone(),
                status: if duration <= case.deadline_s {
                    ReachabilityStatus::Reachable
                } else {
                    ReachabilityStatus::DeadlineExceeded
                },
                reached_safe_state: Some(safe_state),
                worst_case_duration_s: Some(duration),
                path,
                missing_capabilities: Vec::new(),
            });
        }

        let mut missing = BTreeSet::new();
        for edge in &self.transitions {
            if edge.from == case.initial_state {
                for capability in &edge.required_capabilities {
                    if !capabilities.contains(capability) {
                        missing.insert(*capability);
                    }
                }
            }
        }
        Ok(SafeStateReachabilityReport {
            case_id: case.case_id.clone(),
            status: ReachabilityStatus::Unreachable,
            reached_safe_state: None,
            worst_case_duration_s: None,
            path: vec![case.initial_state],
            missing_capabilities: missing.into_iter().collect(),
        })
    }
}

fn transition(
    from: AbstractFlightState,
    to: AbstractFlightState,
    maximum_duration_s: f64,
    required_capabilities: &[SafetyCapability],
    rationale: &str,
) -> SafeStateTransition {
    SafeStateTransition {
        from,
        to,
        maximum_duration_s,
        required_capabilities: required_capabilities.to_vec(),
        rationale: rationale.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powered_case_reaches_ground_safe() {
        let model = SafeStateReachabilityModel::default_helicopter_model();
        let report = model
            .assess(&ReachabilityCase {
                case_id: "powered-return".into(),
                initial_state: AbstractFlightState::NominalMission,
                available_capabilities: vec![
                    SafetyCapability::Navigation,
                    SafetyCapability::MainRotorPower,
                    SafetyCapability::LandingZoneKnown,
                    SafetyCapability::ControlComputer,
                    SafetyCapability::ActuatorPower,
                ],
                deadline_s: 400.0,
            })
            .unwrap();
        assert_eq!(report.status, ReachabilityStatus::Reachable);
        assert_eq!(
            report.reached_safe_state,
            Some(AbstractFlightState::GroundSafe)
        );
    }

    #[test]
    fn engine_loss_can_use_autorotation_path() {
        let model = SafeStateReachabilityModel::default_helicopter_model();
        let report = model
            .assess(&ReachabilityCase {
                case_id: "engine-loss".into(),
                initial_state: AbstractFlightState::DegradedHover,
                available_capabilities: vec![
                    SafetyCapability::RotorEnergy,
                    SafetyCapability::LandingZoneKnown,
                    SafetyCapability::ControlComputer,
                    SafetyCapability::ActuatorPower,
                ],
                deadline_s: 100.0,
            })
            .unwrap();
        assert_eq!(report.status, ReachabilityStatus::Reachable);
        assert!(report.path.contains(&AbstractFlightState::Autorotation));
    }

    #[test]
    fn missing_control_and_rotor_energy_is_unreachable() {
        let model = SafeStateReachabilityModel::default_helicopter_model();
        let report = model
            .assess(&ReachabilityCase {
                case_id: "common-cause-loss".into(),
                initial_state: AbstractFlightState::DegradedHover,
                available_capabilities: vec![SafetyCapability::LandingZoneKnown],
                deadline_s: 100.0,
            })
            .unwrap();
        assert_eq!(report.status, ReachabilityStatus::Unreachable);
        assert!(!report.missing_capabilities.is_empty());
    }

    #[test]
    fn feasible_path_can_still_violate_deadline() {
        let model = SafeStateReachabilityModel::default_helicopter_model();
        let report = model
            .assess(&ReachabilityCase {
                case_id: "too-late".into(),
                initial_state: AbstractFlightState::ReturnToBase,
                available_capabilities: vec![
                    SafetyCapability::Navigation,
                    SafetyCapability::MainRotorPower,
                    SafetyCapability::LandingZoneKnown,
                    SafetyCapability::ActuatorPower,
                ],
                deadline_s: 10.0,
            })
            .unwrap();
        assert_eq!(report.status, ReachabilityStatus::DeadlineExceeded);
    }
}
