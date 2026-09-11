// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integrated adversarial EVA campaign for SX-020.
//!
//! This campaign composes existing simulation twins and decision layers. It is
//! intentionally not a new monolithic physics solver. Cross-subsystem outputs
//! remain explicit so future higher-fidelity models can replace individual
//! assumptions without changing the campaign authority/evidence semantics.

use serde::{Deserialize, Serialize};

use crate::durability_coupling::{
    durability_recommendation, DurabilityMissionModifiers, DurabilityOperationalPolicy,
    DurabilityRecommendation,
};
use crate::dust::{DustDegradationTwin, DustExposure};
use crate::eva_mission::{
    EvaMissionDisposition, EvaMissionPhase, EvaMissionSegment, IntegratedEvaMission,
};
use crate::glove_dust::{
    apply_glove_dust_report, glove_dust_recommendation, GloveDustExposure, GloveDustRecommendation,
    GloveDustTwin,
};
use crate::plss::PlssPathState;
use crate::powered_glove::{
    PoweredGloveCommand, PoweredGloveFailState, PoweredGloveMode, PoweredGloveTwin,
    NUM_GLOVE_DIGITS,
};
use crate::safe_haven::{
    recommend_safe_haven, SafeHavenCandidate, SafeHavenKind, SafeHavenPolicy,
    SafeHavenRecommendation, SafeHavenSituation,
};
use crate::space_exosuit::ExosuitEvidenceLevel;
use crate::suit_service_fault_campaign::{run_service_fault_case, SuitServiceFaultCase};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegratedEvaScenario {
    Nominal,
    DustAndDexterityDegradation,
    MobilityPowerShortfall,
    PrimaryOxygenPathLoss,
    SolarParticleEventCommBlackout,
    GlovePowerLoss,
    ReturnNeededServiceAndShelterUnavailable,
}

impl IntegratedEvaScenario {
    pub const ALL: [Self; 7] = [
        Self::Nominal,
        Self::DustAndDexterityDegradation,
        Self::MobilityPowerShortfall,
        Self::PrimaryOxygenPathLoss,
        Self::SolarParticleEventCommBlackout,
        Self::GlovePowerLoss,
        Self::ReturnNeededServiceAndShelterUnavailable,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntegratedEvaDisposition {
    Continue,
    Degrade,
    ReturnToSafeHaven,
    ImmediateShelter,
    NoFeasibleSafeHaven,
    Abort,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegratedEvaScenarioReport {
    pub scenario: IntegratedEvaScenario,
    pub disposition: IntegratedEvaDisposition,
    pub mission_disposition: EvaMissionDisposition,
    pub durability_recommendation: DurabilityRecommendation,
    pub glove_recommendation: GloveDustRecommendation,
    pub glove_fail_state: PoweredGloveFailState,
    pub glove_backdrivable: bool,
    pub safe_haven_recommendation: Option<SafeHavenRecommendation>,
    pub service_available: Option<bool>,
    pub oxygen_consumed_l: f64,
    pub mobility_energy_wh: f64,
    pub cumulative_radiation_msv: f64,
    pub approximation_notes: Vec<String>,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn run_integrated_eva_scenario(
    scenario: IntegratedEvaScenario,
) -> IntegratedEvaScenarioReport {
    let mut mission = IntegratedEvaMission::simulation_reference();
    let mut segment = EvaMissionSegment::lunar_reference(
        "integrated-surface-task",
        EvaMissionPhase::SurfaceWork,
        600.0,
    );
    segment.personal_dose_rate_msv_h = 0.02;
    segment.forecast_upper_rate_msv_h = 0.03;
    segment.safe_haven_time_min = 10.0;

    let mut suit_dust = DustDegradationTwin::simulation_reference();
    let mut glove_dust = GloveDustTwin::simulation_reference();
    let mut glove = PoweredGloveTwin::simulation_reference();

    let mut dust_deposition = 0.0;
    let mut dust_cycles = 0.0;
    let mut glove_deposition = 0.0;
    let mut glove_cycles = 0.0;
    let mut safe_haven_recommendation = None;
    let mut service_available = None;
    let mut approximation_notes = vec![
        "Radiator derating is represented in durability recommendation; the current IntegratedEvaMission step does not yet inject that fraction into its internal PLSS step.".into(),
        "Glove hand-work is tracked by the glove twin but is not yet added to whole-body metabolic workload.".into(),
    ];

    match scenario {
        IntegratedEvaScenario::Nominal => {}
        IntegratedEvaScenario::DustAndDexterityDegradation => {
            dust_deposition = 4.0;
            dust_cycles = 300.0;
            glove_deposition = 5.0;
            glove_cycles = 300.0;
        }
        IntegratedEvaScenario::MobilityPowerShortfall => {
            mission
                .power_mut_for_fault_injection()
                .config_mut_for_fault_injection()
                .mobility
                .max_discharge_w = 10.0;
        }
        IntegratedEvaScenario::PrimaryOxygenPathLoss => {
            mission
                .plss_mut_for_fault_injection()
                .state_mut_for_fault_injection()
                .primary_oxygen = PlssPathState::Failed;
        }
        IntegratedEvaScenario::SolarParticleEventCommBlackout => {
            segment.energetic_particle_alert = true;
            let mut situation = nominal_safe_haven_situation();
            situation.energetic_particle_alert = true;
            let decision = recommend_safe_haven(
                situation,
                &[nominal_safe_haven("local-storm-rover", 8.0)],
                SafeHavenPolicy::simulation_reference(),
            );
            safe_haven_recommendation = Some(decision.recommendation);
            approximation_notes.push(
                "Communication blackout is represented by relying exclusively on fresh local safe-haven evidence.".into(),
            );
        }
        IntegratedEvaScenario::GlovePowerLoss => {
            glove.state_mut_for_fault_injection().power_available = false;
        }
        IntegratedEvaScenario::ReturnNeededServiceAndShelterUnavailable => {
            dust_deposition = 12.0;
            dust_cycles = 1_000.0;
            glove_deposition = 12.0;
            glove_cycles = 1_000.0;
            let service = run_service_fault_case(SuitServiceFaultCase::ServiceNodePowerLoss);
            service_available = service.service_admitted;

            let mut situation = nominal_safe_haven_situation();
            situation.energetic_particle_alert = true;
            situation.base_is_radiation_shelter = false;
            situation.independent_suit_endurance_min = 30.0;
            situation.required_reserve_min = 10.0;
            let unavailable = SafeHavenCandidate {
                node_id: "unavailable-rover".into(),
                kind: SafeHavenKind::PressurizedRover,
                available: false,
                identity_verified: true,
                radiation_shelter: true,
                travel_time_upper_min: 5.0,
                travel_time_nominal_min: 4.0,
                route_confidence: 0.99,
                observation_age_s: 1.0,
                evidence: ExosuitEvidenceLevel::Simulation,
            };
            let decision = recommend_safe_haven(
                situation,
                &[unavailable],
                SafeHavenPolicy::simulation_reference(),
            );
            safe_haven_recommendation = Some(decision.recommendation);
        }
    }

    let mut dust_exposure = DustExposure::uniform(dust_deposition, segment.duration_s);
    dust_exposure.abrasive_cycles = [dust_cycles; crate::dust::DUST_ZONE_COUNT];
    let dust_report = suit_dust
        .step(dust_exposure)
        .expect("campaign dust exposure must be valid");
    let durability_modifiers = DurabilityMissionModifiers::from_dust(&dust_report)
        .expect("campaign durability modifiers must be valid");
    let durability_recommendation = durability_recommendation(
        durability_modifiers,
        DurabilityOperationalPolicy::simulation_reference(),
    );
    segment = durability_modifiers
        .adjust_segment(&segment)
        .expect("campaign segment must remain valid after durability modifiers");

    let glove_dust_report = glove_dust
        .step(GloveDustExposure::uniform(
            glove_deposition,
            glove_cycles,
            segment.duration_s,
        ))
        .expect("campaign glove dust exposure must be valid");
    apply_glove_dust_report(glove.state_mut_for_fault_injection(), &glove_dust_report);
    let glove_recommendation = glove_dust_recommendation(&glove_dust_report);
    let glove_step = glove
        .step(
            PoweredGloveCommand {
                mode: PoweredGloveMode::GripAssist,
                requested_contact_force_n: [20.0; NUM_GLOVE_DIGITS],
                tendon_speed_m_s: [0.02; NUM_GLOVE_DIGITS],
                exercise_resistance_fraction: 0.0,
            },
            60.0,
        )
        .expect("campaign glove step must remain structurally valid");

    // Account glove/EDS electronics as mission load without folding hand work
    // into the whole-body metabolic model yet.
    segment.mission_power_w += glove_step.electrical_power_w + glove_dust_report.eds_power_w;

    let mission_report = mission.run(&[segment]);

    let mut disposition = map_mission_disposition(mission_report.disposition);
    disposition = disposition.max(map_durability(durability_recommendation));
    disposition = disposition.max(map_glove(glove_recommendation));
    if glove_step.fail_state == PoweredGloveFailState::ServiceRequired {
        disposition = disposition.max(IntegratedEvaDisposition::ReturnToSafeHaven);
    }
    if let Some(recommendation) = safe_haven_recommendation {
        disposition = disposition.max(map_safe_haven(recommendation));
    }
    if service_available == Some(false)
        && disposition >= IntegratedEvaDisposition::ReturnToSafeHaven
        && safe_haven_recommendation == Some(SafeHavenRecommendation::NoFeasibleSafeHaven)
    {
        disposition = IntegratedEvaDisposition::NoFeasibleSafeHaven;
    }

    if scenario == IntegratedEvaScenario::PrimaryOxygenPathLoss
        && mission_report.disposition == EvaMissionDisposition::Continue
    {
        approximation_notes.push(
            "Primary O2 path loss remained survivable in this short scenario because the independent secondary path supplied demand.".into(),
        );
    }

    IntegratedEvaScenarioReport {
        scenario,
        disposition,
        mission_disposition: mission_report.disposition,
        durability_recommendation,
        glove_recommendation,
        glove_fail_state: glove_step.fail_state,
        glove_backdrivable: glove_step.backdrivable,
        safe_haven_recommendation,
        service_available,
        oxygen_consumed_l: mission_report.totals.oxygen_consumed_l,
        mobility_energy_wh: mission_report.totals.mobility_energy_wh,
        cumulative_radiation_msv: mission_report.totals.cumulative_radiation_msv,
        approximation_notes,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

fn map_mission_disposition(value: EvaMissionDisposition) -> IntegratedEvaDisposition {
    match value {
        EvaMissionDisposition::Continue => IntegratedEvaDisposition::Continue,
        EvaMissionDisposition::DegradeMission => IntegratedEvaDisposition::Degrade,
        EvaMissionDisposition::ReturnToSafeHaven => IntegratedEvaDisposition::ReturnToSafeHaven,
        EvaMissionDisposition::ImmediateShelter => IntegratedEvaDisposition::ImmediateShelter,
        EvaMissionDisposition::AbortEva => IntegratedEvaDisposition::Abort,
    }
}

fn map_durability(value: DurabilityRecommendation) -> IntegratedEvaDisposition {
    match value {
        DurabilityRecommendation::Continue => IntegratedEvaDisposition::Continue,
        DurabilityRecommendation::DegradeWork => IntegratedEvaDisposition::Degrade,
        DurabilityRecommendation::ReturnForService => IntegratedEvaDisposition::ReturnToSafeHaven,
        DurabilityRecommendation::InvalidState => IntegratedEvaDisposition::Abort,
    }
}

fn map_glove(value: GloveDustRecommendation) -> IntegratedEvaDisposition {
    match value {
        GloveDustRecommendation::Continue => IntegratedEvaDisposition::Continue,
        GloveDustRecommendation::DegradeDexterity => IntegratedEvaDisposition::Degrade,
        GloveDustRecommendation::ReturnForService => IntegratedEvaDisposition::ReturnToSafeHaven,
        GloveDustRecommendation::InvalidState => IntegratedEvaDisposition::Abort,
    }
}

fn map_safe_haven(value: SafeHavenRecommendation) -> IntegratedEvaDisposition {
    match value {
        SafeHavenRecommendation::Continue => IntegratedEvaDisposition::Continue,
        SafeHavenRecommendation::TurnBack | SafeHavenRecommendation::UseNearestSafeHaven => {
            IntegratedEvaDisposition::ReturnToSafeHaven
        }
        SafeHavenRecommendation::ImmediateShelter => IntegratedEvaDisposition::ImmediateShelter,
        SafeHavenRecommendation::NoFeasibleSafeHaven => {
            IntegratedEvaDisposition::NoFeasibleSafeHaven
        }
        SafeHavenRecommendation::InvalidState => IntegratedEvaDisposition::Abort,
    }
}

fn nominal_safe_haven_situation() -> SafeHavenSituation {
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

fn nominal_safe_haven(id: &str, travel_time_upper_min: f64) -> SafeHavenCandidate {
    SafeHavenCandidate {
        node_id: id.into(),
        kind: SafeHavenKind::PressurizedRover,
        available: true,
        identity_verified: true,
        radiation_shelter: true,
        travel_time_upper_min,
        travel_time_nominal_min: 0.8 * travel_time_upper_min,
        route_confidence: 0.99,
        observation_age_s: 2.0,
        evidence: ExosuitEvidenceLevel::Simulation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_scenario_is_replayable() {
        for scenario in IntegratedEvaScenario::ALL {
            let report = run_integrated_eva_scenario(scenario);
            assert_eq!(report.scenario, scenario);
            assert_eq!(report.evidence, ExosuitEvidenceLevel::Simulation);
            assert!(!report.approximation_notes.is_empty());
        }
    }

    #[test]
    fn nominal_scenario_does_not_escalate_to_return_or_abort() {
        let report = run_integrated_eva_scenario(IntegratedEvaScenario::Nominal);
        assert!(report.disposition <= IntegratedEvaDisposition::Degrade);
    }

    #[test]
    fn mobility_power_shortfall_degrades_without_aborting_survival() {
        let report = run_integrated_eva_scenario(IntegratedEvaScenario::MobilityPowerShortfall);
        assert_eq!(report.mission_disposition, EvaMissionDisposition::DegradeMission);
        assert!(report.disposition >= IntegratedEvaDisposition::Degrade);
        assert!(report.oxygen_consumed_l > 0.0);
    }

    #[test]
    fn primary_oxygen_failure_uses_independent_secondary_path() {
        let report = run_integrated_eva_scenario(IntegratedEvaScenario::PrimaryOxygenPathLoss);
        assert_ne!(report.mission_disposition, EvaMissionDisposition::AbortEva);
        assert!(report.oxygen_consumed_l > 0.0);
    }

    #[test]
    fn solar_particle_event_preempts_task_and_selects_local_shelter() {
        let report = run_integrated_eva_scenario(
            IntegratedEvaScenario::SolarParticleEventCommBlackout,
        );
        assert_eq!(report.mission_disposition, EvaMissionDisposition::ImmediateShelter);
        assert_eq!(
            report.safe_haven_recommendation,
            Some(SafeHavenRecommendation::ImmediateShelter)
        );
        assert_eq!(report.oxygen_consumed_l, 0.0);
    }

    #[test]
    fn glove_power_loss_remains_backdrivable() {
        let report = run_integrated_eva_scenario(IntegratedEvaScenario::GlovePowerLoss);
        assert_eq!(report.glove_fail_state, PoweredGloveFailState::PassiveBackdrivable);
        assert!(report.glove_backdrivable);
    }

    #[test]
    fn compound_return_without_service_or_shelter_is_not_hidden() {
        let report = run_integrated_eva_scenario(
            IntegratedEvaScenario::ReturnNeededServiceAndShelterUnavailable,
        );
        assert_eq!(report.service_available, Some(false));
        assert_eq!(
            report.safe_haven_recommendation,
            Some(SafeHavenRecommendation::NoFeasibleSafeHaven)
        );
        assert_eq!(report.disposition, IntegratedEvaDisposition::NoFeasibleSafeHaven);
    }
}
