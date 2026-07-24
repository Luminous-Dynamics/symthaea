// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission-abort corridor evaluation and selection.
//!
//! Declaring an abort is not enough: the selected escape route must remain
//! navigation-observable, terrain-clear, geofence-compliant, fuel-feasible, and
//! connected to a credible terminal state. This module evaluates explicit
//! corridor samples and returns Pass/Fail/Incomplete evidence without inventing
//! missing terrain or landing-zone data.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbortDestinationKind {
    EmergencyLandingZone,
    ReturnBase,
    SafeHold,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AbortCorridorPoint {
    pub position_m: [f64; 3],
    pub terrain_elevation_m: Option<f64>,
    pub inside_allowed_geofence: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbortCorridorCandidate {
    pub corridor_id: String,
    pub destination: AbortDestinationKind,
    pub points: Vec<AbortCorridorPoint>,
    pub required_fuel_kg: f64,
    pub terminal_state_credible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AbortCorridorConfig {
    pub minimum_terrain_clearance_m: f64,
    pub maximum_segment_length_m: f64,
    pub maximum_climb_gradient: f64,
    pub maximum_descent_gradient: f64,
    pub fuel_contingency_fraction: f64,
    pub maximum_corridor_distance_m: f64,
}

impl Default for AbortCorridorConfig {
    fn default() -> Self {
        Self {
            minimum_terrain_clearance_m: 30.0,
            maximum_segment_length_m: 500.0,
            maximum_climb_gradient: 0.35,
            maximum_descent_gradient: 0.45,
            fuel_contingency_fraction: 0.15,
            maximum_corridor_distance_m: 25_000.0,
        }
    }
}

impl AbortCorridorConfig {
    pub fn validate(&self) -> Result<(), AbortCorridorError> {
        if !self.minimum_terrain_clearance_m.is_finite()
            || self.minimum_terrain_clearance_m < 0.0
            || !self.maximum_segment_length_m.is_finite()
            || self.maximum_segment_length_m <= 0.0
            || !self.maximum_climb_gradient.is_finite()
            || self.maximum_climb_gradient <= 0.0
            || !self.maximum_descent_gradient.is_finite()
            || self.maximum_descent_gradient <= 0.0
            || !self.fuel_contingency_fraction.is_finite()
            || !(0.0..1.0).contains(&self.fuel_contingency_fraction)
            || !self.maximum_corridor_distance_m.is_finite()
            || self.maximum_corridor_distance_m <= 0.0
        {
            return Err(AbortCorridorError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbortCorridorStatus {
    Feasible,
    Infeasible,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbortCorridorIssue {
    EmptyIdentity,
    InsufficientPoints,
    NonFiniteGeometry,
    TerrainUnavailable(usize),
    GeofenceUnavailable(usize),
    GeofenceViolation(usize),
    TerrainClearanceViolation(usize),
    SegmentTooLong(usize),
    ClimbGradientExceeded(usize),
    DescentGradientExceeded(usize),
    CorridorTooLong,
    InsufficientFuel,
    TerminalStateNotCredible,
    NavigationUnavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbortCorridorAssessment {
    pub corridor_id: String,
    pub destination: AbortDestinationKind,
    pub status: AbortCorridorStatus,
    pub issues: Vec<AbortCorridorIssue>,
    pub total_distance_m: f64,
    pub minimum_clearance_m: Option<f64>,
    pub required_fuel_with_contingency_kg: f64,
    pub risk_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbortSelection {
    pub selected: Option<AbortCorridorAssessment>,
    pub assessments: Vec<AbortCorridorAssessment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortCorridorError {
    InvalidConfiguration,
}

#[derive(Debug, Clone)]
pub struct MissionAbortCorridorEvaluator {
    config: AbortCorridorConfig,
}

impl MissionAbortCorridorEvaluator {
    pub fn new(config: AbortCorridorConfig) -> Result<Self, AbortCorridorError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn assess(
        &self,
        candidate: &AbortCorridorCandidate,
        available_fuel_kg: f64,
        navigation_usable: bool,
    ) -> Result<AbortCorridorAssessment, AbortCorridorError> {
        self.config.validate()?;
        let mut issues = Vec::new();
        if candidate.corridor_id.trim().is_empty() {
            issues.push(AbortCorridorIssue::EmptyIdentity);
        }
        if candidate.points.len() < 2 {
            issues.push(AbortCorridorIssue::InsufficientPoints);
        }
        if !navigation_usable {
            issues.push(AbortCorridorIssue::NavigationUnavailable);
        }
        if !candidate.required_fuel_kg.is_finite()
            || candidate.required_fuel_kg < 0.0
            || !available_fuel_kg.is_finite()
            || available_fuel_kg < 0.0
            || candidate
                .points
                .iter()
                .flat_map(|point| point.position_m)
                .any(|value| !value.is_finite())
        {
            issues.push(AbortCorridorIssue::NonFiniteGeometry);
        }
        if !candidate.terminal_state_credible {
            issues.push(AbortCorridorIssue::TerminalStateNotCredible);
        }

        let mut total_distance_m = 0.0;
        let mut minimum_clearance_m: Option<f64> = None;
        for (index, point) in candidate.points.iter().enumerate() {
            match point.terrain_elevation_m {
                Some(terrain) if terrain.is_finite() => {
                    let clearance = point.position_m[2] - terrain;
                    minimum_clearance_m = Some(
                        minimum_clearance_m.map_or(clearance, |current| current.min(clearance)),
                    );
                    if clearance < self.config.minimum_terrain_clearance_m {
                        issues.push(AbortCorridorIssue::TerrainClearanceViolation(index));
                    }
                }
                _ => issues.push(AbortCorridorIssue::TerrainUnavailable(index)),
            }
            match point.inside_allowed_geofence {
                Some(true) => {}
                Some(false) => issues.push(AbortCorridorIssue::GeofenceViolation(index)),
                None => issues.push(AbortCorridorIssue::GeofenceUnavailable(index)),
            }
            if index == 0 {
                continue;
            }
            let previous = candidate.points[index - 1];
            let dx = point.position_m[0] - previous.position_m[0];
            let dy = point.position_m[1] - previous.position_m[1];
            let dz = point.position_m[2] - previous.position_m[2];
            let horizontal = (dx * dx + dy * dy).sqrt();
            let distance = (horizontal * horizontal + dz * dz).sqrt();
            total_distance_m += distance;
            if distance > self.config.maximum_segment_length_m {
                issues.push(AbortCorridorIssue::SegmentTooLong(index));
            }
            let gradient = if horizontal > 1.0e-9 {
                dz / horizontal
            } else if dz > 0.0 {
                f64::INFINITY
            } else if dz < 0.0 {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            if gradient > self.config.maximum_climb_gradient {
                issues.push(AbortCorridorIssue::ClimbGradientExceeded(index));
            }
            if gradient < -self.config.maximum_descent_gradient {
                issues.push(AbortCorridorIssue::DescentGradientExceeded(index));
            }
        }
        if total_distance_m > self.config.maximum_corridor_distance_m {
            issues.push(AbortCorridorIssue::CorridorTooLong);
        }
        let required_fuel_with_contingency_kg =
            candidate.required_fuel_kg * (1.0 + self.config.fuel_contingency_fraction);
        if available_fuel_kg < required_fuel_with_contingency_kg {
            issues.push(AbortCorridorIssue::InsufficientFuel);
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                AbortCorridorIssue::TerrainUnavailable(_)
                    | AbortCorridorIssue::GeofenceUnavailable(_)
                    | AbortCorridorIssue::NavigationUnavailable
            )
        });
        let infeasible = issues.iter().any(|issue| {
            matches!(
                issue,
                AbortCorridorIssue::EmptyIdentity
                    | AbortCorridorIssue::InsufficientPoints
                    | AbortCorridorIssue::NonFiniteGeometry
                    | AbortCorridorIssue::GeofenceViolation(_)
                    | AbortCorridorIssue::TerrainClearanceViolation(_)
                    | AbortCorridorIssue::SegmentTooLong(_)
                    | AbortCorridorIssue::ClimbGradientExceeded(_)
                    | AbortCorridorIssue::DescentGradientExceeded(_)
                    | AbortCorridorIssue::CorridorTooLong
                    | AbortCorridorIssue::InsufficientFuel
                    | AbortCorridorIssue::TerminalStateNotCredible
            )
        });
        let status = if infeasible {
            AbortCorridorStatus::Infeasible
        } else if incomplete {
            AbortCorridorStatus::Incomplete
        } else {
            AbortCorridorStatus::Feasible
        };
        let clearance_penalty = minimum_clearance_m
            .map(|clearance| {
                (self.config.minimum_terrain_clearance_m / clearance.max(1.0)).clamp(0.0, 10.0)
            })
            .unwrap_or(10.0);
        let destination_penalty = match candidate.destination {
            AbortDestinationKind::EmergencyLandingZone => 0.0,
            AbortDestinationKind::ReturnBase => 0.2,
            AbortDestinationKind::SafeHold => 0.5,
        };
        let risk_score = total_distance_m / self.config.maximum_corridor_distance_m
            + clearance_penalty
            + destination_penalty
            + issues.len() as f64;

        Ok(AbortCorridorAssessment {
            corridor_id: candidate.corridor_id.clone(),
            destination: candidate.destination,
            status,
            issues,
            total_distance_m,
            minimum_clearance_m,
            required_fuel_with_contingency_kg,
            risk_score,
        })
    }

    pub fn select(
        &self,
        candidates: &[AbortCorridorCandidate],
        available_fuel_kg: f64,
        navigation_usable: bool,
    ) -> Result<AbortSelection, AbortCorridorError> {
        let mut assessments = candidates
            .iter()
            .map(|candidate| self.assess(candidate, available_fuel_kg, navigation_usable))
            .collect::<Result<Vec<_>, _>>()?;
        assessments.sort_by(|left, right| {
            status_rank(left.status)
                .cmp(&status_rank(right.status))
                .then_with(|| left.risk_score.total_cmp(&right.risk_score))
                .then_with(|| left.corridor_id.cmp(&right.corridor_id))
        });
        let selected = assessments
            .iter()
            .find(|assessment| assessment.status == AbortCorridorStatus::Feasible)
            .cloned();
        Ok(AbortSelection {
            selected,
            assessments,
        })
    }
}

const fn status_rank(status: AbortCorridorStatus) -> u8 {
    match status {
        AbortCorridorStatus::Feasible => 0,
        AbortCorridorStatus::Incomplete => 1,
        AbortCorridorStatus::Infeasible => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, altitude_m: f64) -> AbortCorridorCandidate {
        AbortCorridorCandidate {
            corridor_id: id.into(),
            destination: AbortDestinationKind::EmergencyLandingZone,
            points: vec![
                AbortCorridorPoint {
                    position_m: [0.0, 0.0, altitude_m],
                    terrain_elevation_m: Some(0.0),
                    inside_allowed_geofence: Some(true),
                },
                AbortCorridorPoint {
                    position_m: [100.0, 0.0, altitude_m],
                    terrain_elevation_m: Some(0.0),
                    inside_allowed_geofence: Some(true),
                },
            ],
            required_fuel_kg: 2.0,
            terminal_state_credible: true,
        }
    }

    #[test]
    fn clear_corridor_is_feasible() {
        let evaluator = MissionAbortCorridorEvaluator::new(AbortCorridorConfig::default()).unwrap();
        let assessment = evaluator
            .assess(&candidate("clear", 100.0), 10.0, true)
            .unwrap();
        assert_eq!(assessment.status, AbortCorridorStatus::Feasible);
    }

    #[test]
    fn missing_terrain_is_incomplete_not_assumed_safe() {
        let evaluator = MissionAbortCorridorEvaluator::new(AbortCorridorConfig::default()).unwrap();
        let mut route = candidate("unknown", 100.0);
        route.points[1].terrain_elevation_m = None;
        let assessment = evaluator.assess(&route, 10.0, true).unwrap();
        assert_eq!(assessment.status, AbortCorridorStatus::Incomplete);
    }

    #[test]
    fn terrain_collision_is_infeasible() {
        let evaluator = MissionAbortCorridorEvaluator::new(AbortCorridorConfig::default()).unwrap();
        let assessment = evaluator
            .assess(&candidate("low", 10.0), 10.0, true)
            .unwrap();
        assert_eq!(assessment.status, AbortCorridorStatus::Infeasible);
    }

    #[test]
    fn selector_prefers_feasible_low_risk_corridor() {
        let evaluator = MissionAbortCorridorEvaluator::new(AbortCorridorConfig::default()).unwrap();
        let selection = evaluator
            .select(
                &[candidate("far", 60.0), candidate("clear", 120.0)],
                10.0,
                true,
            )
            .unwrap();
        assert_eq!(selection.selected.unwrap().corridor_id, "clear");
    }
}
