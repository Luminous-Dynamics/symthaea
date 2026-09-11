// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing pressure-damage response composition for SX-021.
//!
//! This module constrains planning using pressure-integrity evidence. It does
//! not command pressure hardware, mobility, or habitat interfaces.

use serde::{Deserialize, Serialize};

use crate::pressure_integrity::{PressureIntegrityAssessment, PressureIntegritySeverity};
use crate::safe_haven::{
    recommend_safe_haven, SafeHavenCandidate, SafeHavenDecision, SafeHavenPolicy,
    SafeHavenRecommendation, SafeHavenSituation,
};
use crate::space_exosuit::{ExosuitEvidenceLevel, SuitSafetyState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureResponseAction {
    Continue,
    Monitor,
    ReturnToSafeHaven,
    PatchThenReturn,
    ImmediateEmergency,
    NoFeasibleSafeHaven,
    InvalidState,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureResponsePolicy {
    /// Minimum localization confidence before an emergency patch can be
    /// recommended by the planning layer.
    pub min_patch_localization_confidence: f64,
    /// Minimum predicted time to critical pressure required to spend time on a
    /// patch attempt rather than immediately prioritizing shelter/return.
    pub min_patch_time_margin_s: f64,
    /// Fractional integrity at/below which powered assistance should be denied
    /// through the existing assist supervisor summary.
    pub assist_integrity_floor: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PressureResponsePolicy {
    pub fn simulation_reference() -> Self {
        Self {
            min_patch_localization_confidence: 0.90,
            min_patch_time_margin_s: 300.0,
            assist_integrity_floor: 0.95,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min_patch_localization_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_patch_localization_confidence)
            && self.min_patch_time_margin_s.is_finite()
            && self.min_patch_time_margin_s >= 0.0
            && self.assist_integrity_floor.is_finite()
            && (0.0..=1.0).contains(&self.assist_integrity_floor)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureResponseDecision {
    pub action: PressureResponseAction,
    pub pressure_limited_endurance_min: f64,
    pub patch_candidate: bool,
    pub safe_haven: SafeHavenDecision,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn pressure_limited_situation(
    mut base: SafeHavenSituation,
    assessment: &PressureIntegrityAssessment,
) -> Option<SafeHavenSituation> {
    if !base.is_valid()
        || !assessment.estimated_pressure_pa.is_finite()
        || !assessment.integrity_fraction.is_finite()
        || !(0.0..=1.0).contains(&assessment.integrity_fraction)
    {
        return None;
    }

    if let Some(time_s) = assessment.time_to_critical_s {
        if !time_s.is_finite() || time_s < 0.0 {
            return None;
        }
        base.independent_suit_endurance_min =
            base.independent_suit_endurance_min.min(time_s / 60.0);
    }
    Some(base)
}

/// Apply the conservative integrity summary to the existing powered-assist
/// observation. This only changes an observation consumed by the assist
/// supervisor; it does not actuate pressure or exoskeleton hardware.
pub fn apply_pressure_integrity_to_assist_state(
    suit: &mut SuitSafetyState,
    assessment: &PressureIntegrityAssessment,
) {
    if assessment.integrity_fraction.is_finite() {
        suit.pressure_integrity = suit
            .pressure_integrity
            .min(assessment.integrity_fraction.clamp(0.0, 1.0));
    } else {
        suit.pressure_integrity = 0.0;
    }
}

pub fn recommend_pressure_response(
    assessment: &PressureIntegrityAssessment,
    base_situation: SafeHavenSituation,
    candidates: &[SafeHavenCandidate],
    haven_policy: SafeHavenPolicy,
    response_policy: PressureResponsePolicy,
) -> PressureResponseDecision {
    if !response_policy.is_valid() {
        return invalid_decision(base_situation.evidence);
    }
    let Some(limited) = pressure_limited_situation(base_situation, assessment) else {
        return invalid_decision(base_situation.evidence.min(response_policy.evidence));
    };

    // A pressure clock shorter than the declared survival reserve means the
    // requested reserve is already unattainable. Do not silently relax it.
    let reserve_unattainable = limited.required_reserve_min > limited.independent_suit_endurance_min;
    let mut emergency_situation = limited;
    if !reserve_unattainable && assessment.severity >= PressureIntegritySeverity::ReturnToSafeHaven {
        // Force the safe-haven planner out of nominal "continue work" behavior
        // without changing the planner's own safety semantics.
        emergency_situation.remaining_work_upper_min =
            emergency_situation.independent_suit_endurance_min;
    }

    let safe_haven = if reserve_unattainable {
        SafeHavenDecision {
            recommendation: SafeHavenRecommendation::NoFeasibleSafeHaven,
            selected_node_id: None,
            selected_travel_time_upper_min: None,
            usable_independent_endurance_min: 0.0,
            communication_required: false,
            evidence: limited.evidence.min(assessment.evidence),
        }
    } else {
        recommend_safe_haven(emergency_situation, candidates, haven_policy)
    };

    let patch_candidate = assessment.localized_zone.is_some()
        && assessment.localization_confidence >= response_policy.min_patch_localization_confidence
        && assessment
            .time_to_critical_s
            .is_some_and(|time| time >= response_policy.min_patch_time_margin_s);

    let action = match assessment.severity {
        PressureIntegritySeverity::InvalidState => PressureResponseAction::InvalidState,
        PressureIntegritySeverity::Nominal => match safe_haven.recommendation {
            SafeHavenRecommendation::Continue => PressureResponseAction::Continue,
            SafeHavenRecommendation::InvalidState => PressureResponseAction::InvalidState,
            SafeHavenRecommendation::NoFeasibleSafeHaven => {
                PressureResponseAction::NoFeasibleSafeHaven
            }
            _ => PressureResponseAction::Monitor,
        },
        PressureIntegritySeverity::Monitor => PressureResponseAction::Monitor,
        PressureIntegritySeverity::ReturnToSafeHaven => {
            if safe_haven.recommendation == SafeHavenRecommendation::NoFeasibleSafeHaven {
                PressureResponseAction::NoFeasibleSafeHaven
            } else if patch_candidate {
                PressureResponseAction::PatchThenReturn
            } else {
                PressureResponseAction::ReturnToSafeHaven
            }
        }
        PressureIntegritySeverity::ImmediateEmergency => {
            if safe_haven.recommendation == SafeHavenRecommendation::NoFeasibleSafeHaven {
                PressureResponseAction::NoFeasibleSafeHaven
            } else {
                PressureResponseAction::ImmediateEmergency
            }
        }
    };

    PressureResponseDecision {
        action,
        pressure_limited_endurance_min: limited.independent_suit_endurance_min,
        patch_candidate,
        evidence: assessment
            .evidence
            .min(base_situation.evidence)
            .min(haven_policy.evidence)
            .min(response_policy.evidence),
        safe_haven,
    }
}

fn invalid_decision(evidence: ExosuitEvidenceLevel) -> PressureResponseDecision {
    PressureResponseDecision {
        action: PressureResponseAction::InvalidState,
        pressure_limited_endurance_min: 0.0,
        patch_candidate: false,
        safe_haven: SafeHavenDecision {
            recommendation: SafeHavenRecommendation::InvalidState,
            selected_node_id: None,
            selected_travel_time_upper_min: None,
            usable_independent_endurance_min: 0.0,
            communication_required: false,
            evidence,
        },
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure_integrity::{PressureIntegritySeverity, PressureZone};
    use crate::safe_haven::{SafeHavenKind, SafeHavenPolicy};

    fn assessment(time_to_critical_s: Option<f64>) -> PressureIntegrityAssessment {
        PressureIntegrityAssessment {
            estimated_pressure_pa: 28_000.0,
            pressure_disagreement_fraction: 0.001,
            total_effective_leak_area_m2: 1.0e-7,
            estimated_mass_loss_kg_s: 1.0e-5,
            estimated_pressure_loss_pa_s: 10.0,
            time_to_warning_s: Some(400.0),
            time_to_critical_s,
            localized_zone: Some(PressureZone::LeftGlove),
            localization_confidence: 0.95,
            severity: PressureIntegritySeverity::ReturnToSafeHaven,
            integrity_fraction: 0.80,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    fn situation() -> SafeHavenSituation {
        SafeHavenSituation {
            independent_suit_endurance_min: 120.0,
            required_reserve_min: 2.0,
            return_to_base_upper_min: 8.0,
            remaining_work_upper_min: 20.0,
            base_is_radiation_shelter: true,
            energetic_particle_alert: false,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    fn haven() -> SafeHavenCandidate {
        SafeHavenCandidate {
            node_id: "rover-1".into(),
            kind: SafeHavenKind::PressurizedRover,
            available: true,
            identity_verified: true,
            radiation_shelter: true,
            travel_time_upper_min: 3.0,
            travel_time_nominal_min: 2.0,
            route_confidence: 0.99,
            observation_age_s: 1.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn pressure_clock_can_dominate_plss_endurance() {
        let limited = pressure_limited_situation(situation(), &assessment(Some(600.0))).unwrap();
        assert_eq!(limited.independent_suit_endurance_min, 10.0);
    }

    #[test]
    fn localized_slow_leak_can_recommend_patch_then_return() {
        let decision = recommend_pressure_response(
            &assessment(Some(1_200.0)),
            situation(),
            &[haven()],
            SafeHavenPolicy::simulation_reference(),
            PressureResponsePolicy::simulation_reference(),
        );
        assert_eq!(decision.action, PressureResponseAction::PatchThenReturn);
        assert!(decision.patch_candidate);
    }

    #[test]
    fn leak_clock_shorter_than_reserve_is_not_silently_relaxed() {
        let decision = recommend_pressure_response(
            &assessment(Some(30.0)),
            situation(),
            &[haven()],
            SafeHavenPolicy::simulation_reference(),
            PressureResponsePolicy::simulation_reference(),
        );
        assert_eq!(decision.action, PressureResponseAction::NoFeasibleSafeHaven);
    }

    #[test]
    fn pressure_integrity_can_only_reduce_assist_integrity_observation() {
        let mut suit = SuitSafetyState {
            suit_pressure_pa: 30_000.0,
            oxygen_partial_pressure_pa: 21_000.0,
            co2_partial_pressure_pa: 400.0,
            wearer_core_temperature_k: 310.0,
            assist_battery_soc: 1.0,
            pressure_integrity: 0.90,
            assist_electronics_health: 1.0,
        };
        let mut higher = assessment(None);
        higher.integrity_fraction = 0.95;
        apply_pressure_integrity_to_assist_state(&mut suit, &higher);
        assert_eq!(suit.pressure_integrity, 0.90);

        let mut lower = assessment(None);
        lower.integrity_fraction = 0.40;
        apply_pressure_integrity_to_assist_state(&mut suit, &lower);
        assert_eq!(suit.pressure_integrity, 0.40);
    }
}
