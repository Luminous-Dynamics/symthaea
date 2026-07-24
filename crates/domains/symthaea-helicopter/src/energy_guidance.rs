// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Energy-aware route and contingency guidance.
//!
//! Route feasibility is assessed with explicit cruise, climb, hover, wind,
//! contingency, and landing-reserve terms. Descent never creates fuel credit,
//! and routes with missing terrain or geofence evidence are `Incomplete` rather
//! than assumed safe.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EnergyRouteKind {
    Primary,
    Alternate,
    ReturnToBase,
    EmergencyLanding,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyRouteSegment {
    pub segment_id: String,
    pub distance_m: f64,
    pub climb_m: f64,
    pub descent_m: f64,
    pub expected_headwind_mps: f64,
    pub hover_time_s: f64,
    pub terrain_verified: bool,
    pub geofence_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyRouteCandidate {
    pub route_id: String,
    pub kind: EnergyRouteKind,
    pub segments: Vec<EnergyRouteSegment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyGuidanceConfig {
    pub schema_version: String,
    pub policy_id: String,
    pub cruise_fuel_kg_per_m: f64,
    pub climb_fuel_kg_per_m: f64,
    pub hover_fuel_kg_per_s: f64,
    pub headwind_fraction_per_mps: f64,
    pub contingency_fraction: f64,
    pub minimum_landing_fuel_kg: f64,
    pub maximum_headwind_mps: f64,
    pub maximum_segment_distance_m: f64,
}

impl Default for EnergyGuidanceConfig {
    fn default() -> Self {
        Self {
            schema_version: "symthaea.helicopter.energy-guidance.v1".into(),
            policy_id: "default-energy-policy".into(),
            cruise_fuel_kg_per_m: 0.000_12,
            climb_fuel_kg_per_m: 0.002,
            hover_fuel_kg_per_s: 0.006,
            headwind_fraction_per_mps: 0.02,
            contingency_fraction: 0.2,
            minimum_landing_fuel_kg: 8.0,
            maximum_headwind_mps: 25.0,
            maximum_segment_distance_m: 100_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyRouteStatus {
    Feasible,
    Infeasible,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnergyRouteIssue {
    TerrainEvidenceMissing(String),
    GeofenceEvidenceMissing(String),
    HeadwindLimitExceeded {
        segment_id: String,
        observed_mps: f64,
        maximum_mps: f64,
    },
    FuelReserveInsufficient {
        required_kg: f64,
        available_kg: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyRouteAssessment {
    pub route_id: String,
    pub kind: EnergyRouteKind,
    pub status: EnergyRouteStatus,
    pub cruise_fuel_kg: f64,
    pub climb_fuel_kg: f64,
    pub hover_fuel_kg: f64,
    pub wind_fuel_kg: f64,
    pub contingency_fuel_kg: f64,
    pub landing_reserve_kg: f64,
    pub total_required_fuel_kg: f64,
    pub remaining_fuel_after_landing_kg: f64,
    pub issues: Vec<EnergyRouteIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyGuidanceAction {
    ProceedPrimary,
    Divert,
    ReturnToBase,
    LandAsSoonAsPracticable,
    HoldForCompleteEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyGuidanceDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub action: EnergyGuidanceAction,
    pub selected_route_id: Option<String>,
    pub assessments: Vec<EnergyRouteAssessment>,
}

impl EnergyGuidanceDecision {
    pub fn canonical_json(&self) -> Result<Vec<u8>, EnergyGuidanceError> {
        let mut canonical = self.clone();
        canonical
            .assessments
            .sort_by(|a, b| a.route_id.cmp(&b.route_id));
        for assessment in &mut canonical.assessments {
            assessment.issues.sort_by_key(issue_sort_key);
        }
        serde_json::to_vec(&canonical).map_err(|_| EnergyGuidanceError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, EnergyGuidanceError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnergyGuidanceError {
    InvalidConfiguration,
    InvalidAvailableFuel,
    EmptyCandidates,
    DuplicateRouteId(String),
    InvalidRoute(String),
    InvalidSegment {
        route_id: String,
        segment_id: String,
    },
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct EnergyAwareGuidance {
    config: EnergyGuidanceConfig,
}

impl EnergyAwareGuidance {
    pub fn new(config: EnergyGuidanceConfig) -> Result<Self, EnergyGuidanceError> {
        let nonnegative = [
            config.cruise_fuel_kg_per_m,
            config.climb_fuel_kg_per_m,
            config.hover_fuel_kg_per_s,
            config.headwind_fraction_per_mps,
            config.contingency_fraction,
            config.minimum_landing_fuel_kg,
        ];
        if config.schema_version.trim().is_empty()
            || config.policy_id.trim().is_empty()
            || nonnegative
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !config.maximum_headwind_mps.is_finite()
            || config.maximum_headwind_mps < 0.0
            || !config.maximum_segment_distance_m.is_finite()
            || config.maximum_segment_distance_m <= 0.0
        {
            return Err(EnergyGuidanceError::InvalidConfiguration);
        }
        Ok(Self { config })
    }

    pub fn assess_route(
        &self,
        available_fuel_kg: f64,
        candidate: &EnergyRouteCandidate,
    ) -> Result<EnergyRouteAssessment, EnergyGuidanceError> {
        if !available_fuel_kg.is_finite() || available_fuel_kg < 0.0 {
            return Err(EnergyGuidanceError::InvalidAvailableFuel);
        }
        if candidate.route_id.trim().is_empty() || candidate.segments.is_empty() {
            return Err(EnergyGuidanceError::InvalidRoute(
                candidate.route_id.clone(),
            ));
        }

        let mut cruise_fuel = 0.0;
        let mut climb_fuel = 0.0;
        let mut hover_fuel = 0.0;
        let mut wind_fuel = 0.0;
        let mut issues = Vec::new();

        for segment in &candidate.segments {
            let values = [
                segment.distance_m,
                segment.climb_m,
                segment.descent_m,
                segment.expected_headwind_mps,
                segment.hover_time_s,
            ];
            if segment.segment_id.trim().is_empty()
                || values.iter().any(|value| !value.is_finite())
                || segment.distance_m < 0.0
                || segment.distance_m > self.config.maximum_segment_distance_m
                || segment.climb_m < 0.0
                || segment.descent_m < 0.0
                || segment.hover_time_s < 0.0
            {
                return Err(EnergyGuidanceError::InvalidSegment {
                    route_id: candidate.route_id.clone(),
                    segment_id: segment.segment_id.clone(),
                });
            }
            if !segment.terrain_verified {
                issues.push(EnergyRouteIssue::TerrainEvidenceMissing(
                    segment.segment_id.clone(),
                ));
            }
            if !segment.geofence_verified {
                issues.push(EnergyRouteIssue::GeofenceEvidenceMissing(
                    segment.segment_id.clone(),
                ));
            }
            if segment.expected_headwind_mps > self.config.maximum_headwind_mps {
                issues.push(EnergyRouteIssue::HeadwindLimitExceeded {
                    segment_id: segment.segment_id.clone(),
                    observed_mps: segment.expected_headwind_mps,
                    maximum_mps: self.config.maximum_headwind_mps,
                });
            }

            let base_cruise = segment.distance_m * self.config.cruise_fuel_kg_per_m;
            cruise_fuel += base_cruise;
            climb_fuel += segment.climb_m * self.config.climb_fuel_kg_per_m;
            hover_fuel += segment.hover_time_s * self.config.hover_fuel_kg_per_s;
            wind_fuel += base_cruise
                * segment.expected_headwind_mps.max(0.0)
                * self.config.headwind_fraction_per_mps;
            // Descent is intentionally not credited as negative fuel.
        }

        let mission_fuel = cruise_fuel + climb_fuel + hover_fuel + wind_fuel;
        let contingency_fuel = mission_fuel * self.config.contingency_fraction;
        let total_required = mission_fuel + contingency_fuel + self.config.minimum_landing_fuel_kg;
        let remaining = available_fuel_kg - total_required;
        if remaining < 0.0 {
            issues.push(EnergyRouteIssue::FuelReserveInsufficient {
                required_kg: total_required,
                available_kg: available_fuel_kg,
            });
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                EnergyRouteIssue::TerrainEvidenceMissing(_)
                    | EnergyRouteIssue::GeofenceEvidenceMissing(_)
            )
        });
        let failed = issues.iter().any(|issue| {
            matches!(
                issue,
                EnergyRouteIssue::HeadwindLimitExceeded { .. }
                    | EnergyRouteIssue::FuelReserveInsufficient { .. }
            )
        });
        let status = if failed {
            EnergyRouteStatus::Infeasible
        } else if incomplete {
            EnergyRouteStatus::Incomplete
        } else {
            EnergyRouteStatus::Feasible
        };

        Ok(EnergyRouteAssessment {
            route_id: candidate.route_id.clone(),
            kind: candidate.kind,
            status,
            cruise_fuel_kg: cruise_fuel,
            climb_fuel_kg: climb_fuel,
            hover_fuel_kg: hover_fuel,
            wind_fuel_kg: wind_fuel,
            contingency_fuel_kg: contingency_fuel,
            landing_reserve_kg: self.config.minimum_landing_fuel_kg,
            total_required_fuel_kg: total_required,
            remaining_fuel_after_landing_kg: remaining,
            issues,
        })
    }

    pub fn decide(
        &self,
        available_fuel_kg: f64,
        candidates: &[EnergyRouteCandidate],
    ) -> Result<EnergyGuidanceDecision, EnergyGuidanceError> {
        if candidates.is_empty() {
            return Err(EnergyGuidanceError::EmptyCandidates);
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut assessments = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !ids.insert(candidate.route_id.clone()) {
                return Err(EnergyGuidanceError::DuplicateRouteId(
                    candidate.route_id.clone(),
                ));
            }
            assessments.push(self.assess_route(available_fuel_kg, candidate)?);
        }

        let selected = select_route(&assessments);
        let (action, selected_route_id) = if let Some(route) = selected {
            let action = match route.kind {
                EnergyRouteKind::Primary => EnergyGuidanceAction::ProceedPrimary,
                EnergyRouteKind::Alternate => EnergyGuidanceAction::Divert,
                EnergyRouteKind::ReturnToBase => EnergyGuidanceAction::ReturnToBase,
                EnergyRouteKind::EmergencyLanding => EnergyGuidanceAction::LandAsSoonAsPracticable,
            };
            (action, Some(route.route_id.clone()))
        } else if assessments
            .iter()
            .any(|assessment| assessment.status == EnergyRouteStatus::Incomplete)
        {
            (EnergyGuidanceAction::HoldForCompleteEvidence, None)
        } else {
            (EnergyGuidanceAction::LandAsSoonAsPracticable, None)
        };

        Ok(EnergyGuidanceDecision {
            schema_version: self.config.schema_version.clone(),
            policy_id: self.config.policy_id.clone(),
            action,
            selected_route_id,
            assessments,
        })
    }
}

fn select_route(assessments: &[EnergyRouteAssessment]) -> Option<&EnergyRouteAssessment> {
    const PRIORITY: [EnergyRouteKind; 4] = [
        EnergyRouteKind::Primary,
        EnergyRouteKind::Alternate,
        EnergyRouteKind::ReturnToBase,
        EnergyRouteKind::EmergencyLanding,
    ];
    for kind in PRIORITY {
        if let Some(best) = assessments
            .iter()
            .filter(|assessment| {
                assessment.kind == kind && assessment.status == EnergyRouteStatus::Feasible
            })
            .min_by(|a, b| {
                a.total_required_fuel_kg
                    .total_cmp(&b.total_required_fuel_kg)
                    .then_with(|| a.route_id.cmp(&b.route_id))
            })
        {
            return Some(best);
        }
    }
    None
}

fn issue_sort_key(issue: &EnergyRouteIssue) -> String {
    format!("{issue:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(id: &str, kind: EnergyRouteKind) -> EnergyRouteCandidate {
        EnergyRouteCandidate {
            route_id: id.into(),
            kind,
            segments: vec![EnergyRouteSegment {
                segment_id: "leg-1".into(),
                distance_m: 1_000.0,
                climb_m: 50.0,
                descent_m: 50.0,
                expected_headwind_mps: 5.0,
                hover_time_s: 20.0,
                terrain_verified: true,
                geofence_verified: true,
            }],
        }
    }

    #[test]
    fn feasible_primary_is_selected() {
        let guidance = EnergyAwareGuidance::new(EnergyGuidanceConfig::default()).unwrap();
        let decision = guidance
            .decide(
                20.0,
                &[
                    route("alternate", EnergyRouteKind::Alternate),
                    route("primary", EnergyRouteKind::Primary),
                ],
            )
            .unwrap();
        assert_eq!(decision.action, EnergyGuidanceAction::ProceedPrimary);
        assert_eq!(decision.selected_route_id.as_deref(), Some("primary"));
    }

    #[test]
    fn missing_terrain_is_incomplete() {
        let guidance = EnergyAwareGuidance::new(EnergyGuidanceConfig::default()).unwrap();
        let mut candidate = route("primary", EnergyRouteKind::Primary);
        candidate.segments[0].terrain_verified = false;
        let assessment = guidance.assess_route(20.0, &candidate).unwrap();
        assert_eq!(assessment.status, EnergyRouteStatus::Incomplete);
    }

    #[test]
    fn descent_never_creates_fuel_credit() {
        let guidance = EnergyAwareGuidance::new(EnergyGuidanceConfig::default()).unwrap();
        let mut low_descent = route("a", EnergyRouteKind::Primary);
        let mut high_descent = route("b", EnergyRouteKind::Primary);
        low_descent.segments[0].descent_m = 0.0;
        high_descent.segments[0].descent_m = 10_000.0;
        let a = guidance.assess_route(20.0, &low_descent).unwrap();
        let b = guidance.assess_route(20.0, &high_descent).unwrap();
        assert_eq!(a.total_required_fuel_kg, b.total_required_fuel_kg);
    }

    #[test]
    fn insufficient_reserve_rejects_route() {
        let guidance = EnergyAwareGuidance::new(EnergyGuidanceConfig::default()).unwrap();
        let assessment = guidance
            .assess_route(1.0, &route("primary", EnergyRouteKind::Primary))
            .unwrap();
        assert_eq!(assessment.status, EnergyRouteStatus::Infeasible);
    }

    #[test]
    fn decision_digest_is_candidate_order_stable() {
        let guidance = EnergyAwareGuidance::new(EnergyGuidanceConfig::default()).unwrap();
        let first = guidance
            .decide(
                20.0,
                &[
                    route("primary", EnergyRouteKind::Primary),
                    route("alternate", EnergyRouteKind::Alternate),
                ],
            )
            .unwrap()
            .digest_fnv1a64()
            .unwrap();
        let second = guidance
            .decide(
                20.0,
                &[
                    route("alternate", EnergyRouteKind::Alternate),
                    route("primary", EnergyRouteKind::Primary),
                ],
            )
            .unwrap()
            .digest_fnv1a64()
            .unwrap();
        assert_eq!(first, second);
    }
}
