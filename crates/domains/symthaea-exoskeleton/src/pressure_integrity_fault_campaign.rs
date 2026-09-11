// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic SX-021 pressure-integrity failure campaign.

use serde::{Deserialize, Serialize};

use crate::pressure_integrity::{
    LeakLocalizationReading, PatchState, PressureDamageSite, PressureDamageSource,
    PressureIntegrityError, PressureIntegritySeverity, PressureIntegrityTwin,
    PressureSensorReading, PressureZone,
};
use crate::pressure_response::{
    recommend_pressure_response, PressureResponseAction, PressureResponsePolicy,
};
use crate::safe_haven::{SafeHavenCandidate, SafeHavenKind, SafeHavenPolicy, SafeHavenSituation};
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureFaultScenario {
    Nominal,
    MmodPuncture,
    GrowingTear,
    SensorDisagreement,
    StaleLocalization,
    MakeupMasksLeak,
    VerifiedPatch,
    NoReachableSafeHaven,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureFaultCampaignResult {
    pub scenario: PressureFaultScenario,
    pub assessment_severity: Option<PressureIntegritySeverity>,
    pub response_action: Option<PressureResponseAction>,
    pub pressure_error: Option<PressureIntegrityError>,
    pub time_to_critical_s: Option<f64>,
    pub effective_leak_area_m2: Option<f64>,
    pub invariant_preserved: bool,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn run_pressure_fault_campaign() -> Vec<PressureFaultCampaignResult> {
    [
        PressureFaultScenario::Nominal,
        PressureFaultScenario::MmodPuncture,
        PressureFaultScenario::GrowingTear,
        PressureFaultScenario::SensorDisagreement,
        PressureFaultScenario::StaleLocalization,
        PressureFaultScenario::MakeupMasksLeak,
        PressureFaultScenario::VerifiedPatch,
        PressureFaultScenario::NoReachableSafeHaven,
    ]
    .into_iter()
    .map(run_scenario)
    .collect()
}

fn run_scenario(scenario: PressureFaultScenario) -> PressureFaultCampaignResult {
    match scenario {
        PressureFaultScenario::Nominal => nominal(),
        PressureFaultScenario::MmodPuncture => puncture(),
        PressureFaultScenario::GrowingTear => growing_tear(),
        PressureFaultScenario::SensorDisagreement => sensor_disagreement(),
        PressureFaultScenario::StaleLocalization => stale_localization(),
        PressureFaultScenario::MakeupMasksLeak => makeup_masks_leak(),
        PressureFaultScenario::VerifiedPatch => verified_patch(),
        PressureFaultScenario::NoReachableSafeHaven => no_reachable_safe_haven(),
    }
}

fn nominal() -> PressureFaultCampaignResult {
    let twin = PressureIntegrityTwin::simulation_reference();
    let result = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::Nominal,
        assessment_severity: Some(result.severity),
        response_action: None,
        pressure_error: None,
        time_to_critical_s: result.time_to_critical_s,
        effective_leak_area_m2: Some(result.total_effective_leak_area_m2),
        invariant_preserved: result.severity == PressureIntegritySeverity::Nominal
            && result.total_effective_leak_area_m2 == 0.0,
        evidence: result.evidence,
    }
}

fn puncture() -> PressureFaultCampaignResult {
    let mut twin = PressureIntegrityTwin::simulation_reference();
    twin.add_damage_site(damage(PressureZone::Torso, 1.0e-6, 0.0)).unwrap();
    let result = twin.assess(&sensors(30_000.0), &localization(PressureZone::Torso, 0.95, 0.1), 0.0, 295.0, 0.0).unwrap();
    let response = response(&result, &[haven("rover", 3.0)]);
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::MmodPuncture,
        assessment_severity: Some(result.severity),
        response_action: Some(response.action),
        pressure_error: None,
        time_to_critical_s: result.time_to_critical_s,
        effective_leak_area_m2: Some(result.total_effective_leak_area_m2),
        invariant_preserved: result.time_to_critical_s.is_some()
            && !matches!(response.action, PressureResponseAction::Continue),
        evidence: result.evidence,
    }
}

fn growing_tear() -> PressureFaultCampaignResult {
    let mut twin = PressureIntegrityTwin::simulation_reference();
    twin.add_damage_site(damage(PressureZone::LeftArm, 2.0e-7, 0.01)).unwrap();
    let before = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
    twin.propagate_damage(60.0).unwrap();
    let after = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::GrowingTear,
        assessment_severity: Some(after.severity),
        response_action: None,
        pressure_error: None,
        time_to_critical_s: after.time_to_critical_s,
        effective_leak_area_m2: Some(after.total_effective_leak_area_m2),
        invariant_preserved: after.total_effective_leak_area_m2 > before.total_effective_leak_area_m2
            && after.estimated_mass_loss_kg_s > before.estimated_mass_loss_kg_s,
        evidence: after.evidence,
    }
}

fn sensor_disagreement() -> PressureFaultCampaignResult {
    let twin = PressureIntegrityTwin::simulation_reference();
    let readings = [
        PressureSensorReading { pressure_pa: 30_000.0, confidence: 0.99, age_s: 0.1 },
        PressureSensorReading { pressure_pa: 20_000.0, confidence: 0.99, age_s: 0.1 },
    ];
    let error = twin.assess(&readings, &[], 0.0, 295.0, 0.0).unwrap_err();
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::SensorDisagreement,
        assessment_severity: None,
        response_action: Some(PressureResponseAction::InvalidState),
        pressure_error: Some(error),
        time_to_critical_s: None,
        effective_leak_area_m2: None,
        invariant_preserved: error == PressureIntegrityError::ContradictoryPressureSensors,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

fn stale_localization() -> PressureFaultCampaignResult {
    let mut twin = PressureIntegrityTwin::simulation_reference();
    twin.add_damage_site(damage(PressureZone::RightGlove, 2.0e-7, 0.0)).unwrap();
    let result = twin
        .assess(
            &sensors(30_000.0),
            &localization(PressureZone::RightGlove, 0.99, 20.0),
            0.0,
            295.0,
            0.0,
        )
        .unwrap();
    let response = response(&result, &[haven("rover", 3.0)]);
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::StaleLocalization,
        assessment_severity: Some(result.severity),
        response_action: Some(response.action),
        pressure_error: None,
        time_to_critical_s: result.time_to_critical_s,
        effective_leak_area_m2: Some(result.total_effective_leak_area_m2),
        invariant_preserved: result.localized_zone.is_none() && !response.patch_candidate,
        evidence: result.evidence,
    }
}

fn makeup_masks_leak() -> PressureFaultCampaignResult {
    let mut twin = PressureIntegrityTwin::simulation_reference();
    twin.add_damage_site(damage(PressureZone::PlssInterface, 2.0e-7, 0.0)).unwrap();
    let baseline = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
    let result = twin
        .assess(
            &sensors(30_000.0),
            &[],
            0.0,
            295.0,
            baseline.estimated_mass_loss_kg_s * 1.1,
        )
        .unwrap();
    let response = response(&result, &[haven("hab", 4.0)]);
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::MakeupMasksLeak,
        assessment_severity: Some(result.severity),
        response_action: Some(response.action),
        pressure_error: None,
        time_to_critical_s: result.time_to_critical_s,
        effective_leak_area_m2: Some(result.total_effective_leak_area_m2),
        invariant_preserved: result.estimated_pressure_loss_pa_s == 0.0
            && result.total_effective_leak_area_m2 > 0.0
            && result.severity != PressureIntegritySeverity::Nominal,
        evidence: result.evidence,
    }
}

fn verified_patch() -> PressureFaultCampaignResult {
    let mut twin = PressureIntegrityTwin::simulation_reference();
    let index = twin.add_damage_site(damage(PressureZone::LeftGlove, 1.0e-6, 0.0)).unwrap();
    let before = twin
        .assess(
            &sensors(30_000.0),
            &localization(PressureZone::LeftGlove, 0.98, 0.1),
            0.0,
            295.0,
            0.0,
        )
        .unwrap();
    twin.apply_patch(index, PatchState::TemporaryVerified, 0.10, ExosuitEvidenceLevel::Simulation)
        .unwrap();
    let after = twin
        .assess(
            &sensors(30_000.0),
            &localization(PressureZone::LeftGlove, 0.98, 0.1),
            0.0,
            295.0,
            0.0,
        )
        .unwrap();
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::VerifiedPatch,
        assessment_severity: Some(after.severity),
        response_action: None,
        pressure_error: None,
        time_to_critical_s: after.time_to_critical_s,
        effective_leak_area_m2: Some(after.total_effective_leak_area_m2),
        invariant_preserved: after.estimated_mass_loss_kg_s < before.estimated_mass_loss_kg_s
            && after.time_to_critical_s.zip(before.time_to_critical_s).is_some_and(|(a, b)| a > b),
        evidence: after.evidence,
    }
}

fn no_reachable_safe_haven() -> PressureFaultCampaignResult {
    let mut twin = PressureIntegrityTwin::simulation_reference();
    twin.add_damage_site(damage(PressureZone::Torso, 2.0e-6, 0.0)).unwrap();
    let result = twin.assess(&sensors(26_000.0), &[], 0.0, 295.0, 0.0).unwrap();
    let response = response(&result, &[haven("too-far", 90.0)]);
    PressureFaultCampaignResult {
        scenario: PressureFaultScenario::NoReachableSafeHaven,
        assessment_severity: Some(result.severity),
        response_action: Some(response.action),
        pressure_error: None,
        time_to_critical_s: result.time_to_critical_s,
        effective_leak_area_m2: Some(result.total_effective_leak_area_m2),
        invariant_preserved: matches!(
            response.action,
            PressureResponseAction::NoFeasibleSafeHaven | PressureResponseAction::ImmediateEmergency
        ),
        evidence: result.evidence,
    }
}

fn response(
    assessment: &crate::pressure_integrity::PressureIntegrityAssessment,
    havens: &[SafeHavenCandidate],
) -> crate::pressure_response::PressureResponseDecision {
    recommend_pressure_response(
        assessment,
        SafeHavenSituation {
            independent_suit_endurance_min: 120.0,
            required_reserve_min: 2.0,
            return_to_base_upper_min: 15.0,
            remaining_work_upper_min: 20.0,
            base_is_radiation_shelter: true,
            energetic_particle_alert: false,
            evidence: ExosuitEvidenceLevel::Simulation,
        },
        havens,
        SafeHavenPolicy::simulation_reference(),
        PressureResponsePolicy::simulation_reference(),
    )
}

fn sensors(pressure_pa: f64) -> [PressureSensorReading; 3] {
    [
        PressureSensorReading { pressure_pa, confidence: 0.99, age_s: 0.1 },
        PressureSensorReading { pressure_pa: pressure_pa * 1.001, confidence: 0.98, age_s: 0.1 },
        PressureSensorReading { pressure_pa: pressure_pa * 0.999, confidence: 0.97, age_s: 0.2 },
    ]
}

fn localization(zone: PressureZone, confidence: f64, age_s: f64) -> [LeakLocalizationReading; 1] {
    [LeakLocalizationReading {
        zone,
        anomaly_score: 0.95,
        confidence,
        age_s,
    }]
}

fn damage(zone: PressureZone, area_m2: f64, growth_rate_s: f64) -> PressureDamageSite {
    PressureDamageSite {
        zone,
        source: PressureDamageSource::MmodImpact,
        effective_orifice_area_m2: area_m2,
        fractional_growth_rate_s: growth_rate_s,
        patch_residual_fraction: 1.0,
        patch_state: PatchState::None,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

fn haven(id: &str, travel_min: f64) -> SafeHavenCandidate {
    SafeHavenCandidate {
        node_id: id.into(),
        kind: SafeHavenKind::PressurizedRover,
        available: true,
        identity_verified: true,
        radiation_shelter: true,
        travel_time_upper_min: travel_min,
        travel_time_nominal_min: 0.8 * travel_min,
        route_confidence: 0.99,
        observation_age_s: 1.0,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_pressure_fault_scenarios_preserve_declared_containment_invariants() {
        let results = run_pressure_fault_campaign();
        assert_eq!(results.len(), 8);
        assert!(results.iter().all(|result| result.invariant_preserved), "{results:#?}");
    }
}
