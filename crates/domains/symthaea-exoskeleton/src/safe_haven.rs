// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing safe-haven recommendation for SX-019.
//!
//! This module never drives locomotion. It determines whether a locally known
//! rover/habitat/shelter is feasible using independent suit endurance, route
//! uncertainty, local evidence freshness, and radiation shelter capability.
//! Communications are not required when local evidence is fresh enough.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafeHavenKind {
    Habitat,
    PressurizedRover,
    FixedStormShelter,
    Lander,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeHavenCandidate {
    pub node_id: String,
    pub kind: SafeHavenKind,
    pub available: bool,
    pub identity_verified: bool,
    pub radiation_shelter: bool,
    /// Conservative route-time estimate used for admission, minutes.
    pub travel_time_upper_min: f64,
    /// Nominal route estimate, minutes.
    pub travel_time_nominal_min: f64,
    /// Confidence in local route/state knowledge [0,1].
    pub route_confidence: f64,
    /// Age of local node/route evidence, seconds.
    pub observation_age_s: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl SafeHavenCandidate {
    pub fn is_valid(&self) -> bool {
        !self.node_id.trim().is_empty()
            && self.travel_time_upper_min.is_finite()
            && self.travel_time_upper_min >= 0.0
            && self.travel_time_nominal_min.is_finite()
            && self.travel_time_nominal_min >= 0.0
            && self.travel_time_nominal_min <= self.travel_time_upper_min
            && self.route_confidence.is_finite()
            && (0.0..=1.0).contains(&self.route_confidence)
            && self.observation_age_s.is_finite()
            && self.observation_age_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SafeHavenSituation {
    /// Conservative independently available suit survival endurance, minutes.
    pub independent_suit_endurance_min: f64,
    /// Endurance deliberately held back after arrival, minutes.
    pub required_reserve_min: f64,
    /// Conservative time to return to the nominal base/habitat, minutes.
    pub return_to_base_upper_min: f64,
    /// Remaining planned work before normal return, conservative minutes.
    pub remaining_work_upper_min: f64,
    pub base_is_radiation_shelter: bool,
    pub energetic_particle_alert: bool,
    pub evidence: ExosuitEvidenceLevel,
}

impl SafeHavenSituation {
    pub fn is_valid(&self) -> bool {
        [
            self.independent_suit_endurance_min,
            self.required_reserve_min,
            self.return_to_base_upper_min,
            self.remaining_work_upper_min,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
            && self.required_reserve_min <= self.independent_suit_endurance_min
    }

    pub fn usable_endurance_min(&self) -> f64 {
        (self.independent_suit_endurance_min - self.required_reserve_min).max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SafeHavenPolicy {
    pub min_route_confidence: f64,
    pub max_observation_age_s: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl SafeHavenPolicy {
    pub fn simulation_reference() -> Self {
        Self {
            min_route_confidence: 0.90,
            max_observation_age_s: 30.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min_route_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_route_confidence)
            && self.max_observation_age_s.is_finite()
            && self.max_observation_age_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafeHavenRecommendation {
    Continue,
    TurnBack,
    UseNearestSafeHaven,
    ImmediateShelter,
    NoFeasibleSafeHaven,
    InvalidState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeHavenDecision {
    pub recommendation: SafeHavenRecommendation,
    pub selected_node_id: Option<String>,
    pub selected_travel_time_upper_min: Option<f64>,
    pub usable_independent_endurance_min: f64,
    pub communication_required: bool,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn recommend_safe_haven(
    situation: SafeHavenSituation,
    candidates: &[SafeHavenCandidate],
    policy: SafeHavenPolicy,
) -> SafeHavenDecision {
    if !situation.is_valid()
        || !policy.is_valid()
        || candidates.iter().any(|candidate| !candidate.is_valid())
    {
        return decision(
            SafeHavenRecommendation::InvalidState,
            None,
            situation.usable_endurance_min(),
            situation.evidence,
        );
    }

    let usable = situation.usable_endurance_min();
    let require_radiation_shelter = situation.energetic_particle_alert;
    let best = candidates
        .iter()
        .filter(|candidate| {
            candidate.available
                && candidate.identity_verified
                && candidate.route_confidence >= policy.min_route_confidence
                && candidate.observation_age_s <= policy.max_observation_age_s
                && candidate.travel_time_upper_min <= usable
                && (!require_radiation_shelter || candidate.radiation_shelter)
        })
        .min_by(|a, b| {
            a.travel_time_upper_min
                .total_cmp(&b.travel_time_upper_min)
        });

    if situation.energetic_particle_alert {
        if let Some(candidate) = best {
            return decision(
                SafeHavenRecommendation::ImmediateShelter,
                Some(candidate),
                usable,
                situation.evidence.min(candidate.evidence),
            );
        }
        if situation.base_is_radiation_shelter && situation.return_to_base_upper_min <= usable {
            return SafeHavenDecision {
                recommendation: SafeHavenRecommendation::ImmediateShelter,
                selected_node_id: Some("nominal-base".into()),
                selected_travel_time_upper_min: Some(situation.return_to_base_upper_min),
                usable_independent_endurance_min: usable,
                communication_required: false,
                evidence: situation.evidence,
            };
        }
        return decision(
            SafeHavenRecommendation::NoFeasibleSafeHaven,
            None,
            usable,
            situation.evidence,
        );
    }

    let nominal_plan_time = situation.remaining_work_upper_min + situation.return_to_base_upper_min;
    if nominal_plan_time <= usable {
        return decision(
            SafeHavenRecommendation::Continue,
            None,
            usable,
            situation.evidence,
        );
    }

    if let Some(candidate) = best {
        return decision(
            SafeHavenRecommendation::UseNearestSafeHaven,
            Some(candidate),
            usable,
            situation.evidence.min(candidate.evidence),
        );
    }

    if situation.return_to_base_upper_min <= usable {
        return SafeHavenDecision {
            recommendation: SafeHavenRecommendation::TurnBack,
            selected_node_id: Some("nominal-base".into()),
            selected_travel_time_upper_min: Some(situation.return_to_base_upper_min),
            usable_independent_endurance_min: usable,
            communication_required: false,
            evidence: situation.evidence,
        };
    }

    decision(
        SafeHavenRecommendation::NoFeasibleSafeHaven,
        None,
        usable,
        situation.evidence,
    )
}

fn decision(
    recommendation: SafeHavenRecommendation,
    candidate: Option<&SafeHavenCandidate>,
    usable: f64,
    evidence: ExosuitEvidenceLevel,
) -> SafeHavenDecision {
    SafeHavenDecision {
        recommendation,
        selected_node_id: candidate.map(|value| value.node_id.clone()),
        selected_travel_time_upper_min: candidate.map(|value| value.travel_time_upper_min),
        usable_independent_endurance_min: usable,
        communication_required: false,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn situation() -> SafeHavenSituation {
        SafeHavenSituation {
            independent_suit_endurance_min: 120.0,
            required_reserve_min: 20.0,
            return_to_base_upper_min: 40.0,
            remaining_work_upper_min: 30.0,
            base_is_radiation_shelter: true,
            energetic_particle_alert: false,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    fn candidate(id: &str, travel: f64) -> SafeHavenCandidate {
        SafeHavenCandidate {
            node_id: id.into(),
            kind: SafeHavenKind::PressurizedRover,
            available: true,
            identity_verified: true,
            radiation_shelter: true,
            travel_time_upper_min: travel,
            travel_time_nominal_min: 0.8 * travel,
            route_confidence: 0.98,
            observation_age_s: 5.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn nominal_plan_continues_when_independent_endurance_is_sufficient() {
        let result = recommend_safe_haven(
            situation(),
            &[candidate("rover", 10.0)],
            SafeHavenPolicy::simulation_reference(),
        );
        assert_eq!(result.recommendation, SafeHavenRecommendation::Continue);
    }

    #[test]
    fn solar_particle_alert_selects_feasible_radiation_shelter_offline() {
        let mut emergency = situation();
        emergency.energetic_particle_alert = true;
        let result = recommend_safe_haven(
            emergency,
            &[candidate("storm-rover", 8.0)],
            SafeHavenPolicy::simulation_reference(),
        );
        assert_eq!(result.recommendation, SafeHavenRecommendation::ImmediateShelter);
        assert_eq!(result.selected_node_id.as_deref(), Some("storm-rover"));
        assert!(!result.communication_required);
    }

    #[test]
    fn stale_rover_is_not_treated_as_safe_haven() {
        let mut long_plan = situation();
        long_plan.remaining_work_upper_min = 90.0;
        let mut stale = candidate("stale-rover", 10.0);
        stale.observation_age_s = 120.0;
        let result = recommend_safe_haven(
            long_plan,
            &[stale],
            SafeHavenPolicy::simulation_reference(),
        );
        assert_eq!(result.recommendation, SafeHavenRecommendation::TurnBack);
    }

    #[test]
    fn unreachable_shelters_are_not_selected() {
        let mut emergency = situation();
        emergency.energetic_particle_alert = true;
        emergency.base_is_radiation_shelter = false;
        emergency.independent_suit_endurance_min = 30.0;
        emergency.required_reserve_min = 10.0;
        let result = recommend_safe_haven(
            emergency,
            &[candidate("too-far", 25.0)],
            SafeHavenPolicy::simulation_reference(),
        );
        assert_eq!(
            result.recommendation,
            SafeHavenRecommendation::NoFeasibleSafeHaven
        );
    }
}
