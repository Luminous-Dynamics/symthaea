// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SX-022 cross-domain fidelity coupling for integrated EVA simulation.
//!
//! This thin layer closes two previously explicit approximation gaps:
//! radiator durability evidence now reaches the PLSS thermal step, and powered
//! glove hand effort reaches whole-body mission workload. It remains a
//! simulation composition layer and commands no physical hardware.

use serde::{Deserialize, Serialize};

use crate::durability_coupling::DurabilityMissionModifiers;
use crate::eva_mission::{EvaMissionSegment, EvaMissionSegmentReport, IntegratedEvaMission};
use crate::powered_glove::{PoweredGloveCommand, PoweredGloveStep, NUM_GLOVE_DIGITS};
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvaFidelityCouplingConfig {
    /// For near-static grip, convert sustained wearer force into a mechanical-
    /// equivalent workload passed to the existing metabolism model, W/N.
    /// This is a simulation coefficient, not validated hand physiology.
    pub isometric_hand_equivalent_w_per_n: f64,
    /// Below this tendon speed, a digit is treated as near-isometric, m/s.
    pub isometric_speed_threshold_m_s: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl EvaFidelityCouplingConfig {
    pub fn simulation_reference() -> Self {
        Self {
            isometric_hand_equivalent_w_per_n: 0.08,
            isometric_speed_threshold_m_s: 0.002,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.isometric_hand_equivalent_w_per_n.is_finite()
            && self.isometric_hand_equivalent_w_per_n >= 0.0
            && self.isometric_speed_threshold_m_s.is_finite()
            && self.isometric_speed_threshold_m_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaFidelityError {
    InvalidConfig,
    InvalidBaseSegment,
    InvalidDurabilityModifiers,
    InvalidGloveCommand,
    InvalidGloveStep,
    InvalidPlssDerating,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaFidelityCoupledSegment {
    pub segment: EvaMissionSegment,
    pub radiator_heat_rejection_fraction: f64,
    pub dynamic_hand_mechanical_power_w: f64,
    pub isometric_hand_equivalent_power_w: f64,
    pub total_hand_equivalent_power_w: f64,
    pub glove_electrical_power_w: f64,
    pub glove_actuator_mechanical_power_w: f64,
    pub glove_waste_heat_w: f64,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn compose_fidelity_segment(
    base: &EvaMissionSegment,
    durability: DurabilityMissionModifiers,
    glove_command: &PoweredGloveCommand,
    glove_step: &PoweredGloveStep,
    config: EvaFidelityCouplingConfig,
) -> Result<EvaFidelityCoupledSegment, EvaFidelityError> {
    if !config.is_valid() {
        return Err(EvaFidelityError::InvalidConfig);
    }
    if !base.is_valid() {
        return Err(EvaFidelityError::InvalidBaseSegment);
    }
    if !durability.is_valid() {
        return Err(EvaFidelityError::InvalidDurabilityModifiers);
    }
    if !glove_command.is_valid() {
        return Err(EvaFidelityError::InvalidGloveCommand);
    }
    if !valid_glove_step(glove_step) {
        return Err(EvaFidelityError::InvalidGloveStep);
    }

    let mut segment = durability
        .adjust_segment(base)
        .ok_or(EvaFidelityError::InvalidDurabilityModifiers)?;

    let body_assist_requested_w =
        segment.gross_positive_mechanical_power_w * segment.requested_assist_fraction;
    let (dynamic_hand_w, isometric_hand_w) =
        hand_work_equivalent(glove_command, glove_step, config);
    let total_hand_w = dynamic_hand_w + isometric_hand_w;

    let glove_actuator_mechanical_w = (0..NUM_GLOVE_DIGITS)
        .map(|i| {
            (glove_step.assist_force_n[i] + glove_step.exercise_resistance_force_n[i])
                * glove_command.tendon_speed_m_s[i]
        })
        .sum::<f64>();
    let glove_waste_heat_w =
        (glove_step.electrical_power_w - glove_actuator_mechanical_w).max(0.0);

    segment.gross_positive_mechanical_power_w += total_hand_w;
    segment.requested_assist_fraction = if segment.gross_positive_mechanical_power_w > 0.0 {
        (body_assist_requested_w / segment.gross_positive_mechanical_power_w).clamp(0.0, 1.0)
    } else {
        0.0
    };
    segment.mobility_overhead_w += glove_step.electrical_power_w;
    segment.non_actuator_equipment_heat_w += glove_waste_heat_w;
    segment.evidence = segment
        .evidence
        .min(durability.evidence)
        .min(glove_step.evidence)
        .min(config.evidence);

    if !segment.is_valid() {
        return Err(EvaFidelityError::InvalidBaseSegment);
    }

    Ok(EvaFidelityCoupledSegment {
        segment,
        radiator_heat_rejection_fraction: durability.radiator_heat_rejection_fraction,
        dynamic_hand_mechanical_power_w: dynamic_hand_w,
        isometric_hand_equivalent_power_w: isometric_hand_w,
        total_hand_equivalent_power_w: total_hand_w,
        glove_electrical_power_w: glove_step.electrical_power_w,
        glove_actuator_mechanical_power_w,
        glove_waste_heat_w,
        evidence: base
            .evidence
            .min(durability.evidence)
            .min(glove_step.evidence)
            .min(config.evidence),
    })
}

pub fn step_fidelity_segment(
    mission: &mut IntegratedEvaMission,
    coupled: &EvaFidelityCoupledSegment,
) -> Result<EvaMissionSegmentReport, EvaFidelityError> {
    if !coupled.radiator_heat_rejection_fraction.is_finite()
        || !(0.0..=1.0).contains(&coupled.radiator_heat_rejection_fraction)
    {
        return Err(EvaFidelityError::InvalidPlssDerating);
    }

    let previous = mission
        .plss_mut_for_fault_injection()
        .environment_heat_rejection_fraction();
    mission
        .plss_mut_for_fault_injection()
        .set_environment_heat_rejection_fraction(coupled.radiator_heat_rejection_fraction)
        .map_err(|_| EvaFidelityError::InvalidPlssDerating)?;

    let report = mission.step(&coupled.segment);

    mission
        .plss_mut_for_fault_injection()
        .set_environment_heat_rejection_fraction(previous)
        .map_err(|_| EvaFidelityError::InvalidPlssDerating)?;

    Ok(report)
}

fn hand_work_equivalent(
    command: &PoweredGloveCommand,
    step: &PoweredGloveStep,
    config: EvaFidelityCouplingConfig,
) -> (f64, f64) {
    let mut dynamic = 0.0;
    let mut isometric = 0.0;
    for i in 0..NUM_GLOVE_DIGITS {
        let force = step.human_required_force_n[i].max(0.0);
        let speed = command.tendon_speed_m_s[i].max(0.0);
        if speed <= config.isometric_speed_threshold_m_s {
            let intentionally_loaded = command.requested_contact_force_n[i] > 0.0
                || step.exercise_resistance_force_n[i] > 0.0;
            if intentionally_loaded {
                isometric += force * config.isometric_hand_equivalent_w_per_n;
            }
        } else {
            dynamic += force * speed;
        }
    }
    (dynamic, isometric)
}

fn valid_glove_step(step: &PoweredGloveStep) -> bool {
    step.pressure_resistance_force_n
        .iter()
        .chain(step.exercise_resistance_force_n.iter())
        .chain(step.assist_force_n.iter())
        .chain(step.human_required_force_n.iter())
        .chain(step.tactile_fidelity.iter())
        .chain(step.fatigue.iter())
        .all(|value| value.is_finite() && *value >= 0.0)
        && step.electrical_power_w.is_finite()
        && step.electrical_power_w >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::durability_coupling::DurabilityMissionModifiers;
    use crate::eva_mission::EvaMissionPhase;
    use crate::powered_glove::{PoweredGloveCommand, PoweredGloveMode, PoweredGloveTwin};

    fn base_segment() -> EvaMissionSegment {
        EvaMissionSegment::lunar_reference("coupled", EvaMissionPhase::SurfaceWork, 60.0)
    }

    fn command(mode: PoweredGloveMode, speed: f64) -> PoweredGloveCommand {
        PoweredGloveCommand {
            mode,
            requested_contact_force_n: [20.0; NUM_GLOVE_DIGITS],
            tendon_speed_m_s: [speed; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
        }
    }

    #[test]
    fn dynamic_hand_work_enters_whole_body_metabolic_path_without_body_assist_claiming_it() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        let cmd = command(PoweredGloveMode::Transparent, 0.02);
        let step = glove.step(cmd, 1.0).unwrap();
        let coupled = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &cmd,
            &step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();
        assert!(coupled.dynamic_hand_mechanical_power_w > 0.0);
        assert!(
            coupled.segment.gross_positive_mechanical_power_w
                > base_segment().gross_positive_mechanical_power_w
        );
        let original_assist = base_segment().gross_positive_mechanical_power_w
            * base_segment().requested_assist_fraction;
        let coupled_assist = coupled.segment.gross_positive_mechanical_power_w
            * coupled.segment.requested_assist_fraction;
        assert!((original_assist - coupled_assist).abs() < 1e-9);
    }

    #[test]
    fn powered_glove_assist_reduces_hand_equivalent_work_but_consumes_mobility_power() {
        let cmd_transparent = command(PoweredGloveMode::Transparent, 0.02);
        let cmd_assisted = command(PoweredGloveMode::GripAssist, 0.02);
        let mut transparent = PoweredGloveTwin::simulation_reference();
        let mut assisted = PoweredGloveTwin::simulation_reference();
        let a_step = transparent.step(cmd_transparent, 1.0).unwrap();
        let b_step = assisted.step(cmd_assisted, 1.0).unwrap();
        let a = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &cmd_transparent,
            &a_step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();
        let b = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &cmd_assisted,
            &b_step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();
        assert!(b.total_hand_equivalent_power_w < a.total_hand_equivalent_power_w);
        assert!(b.glove_electrical_power_w > a.glove_electrical_power_w);
        assert!(b.segment.mobility_overhead_w > a.segment.mobility_overhead_w);
    }

    #[test]
    fn radiator_derating_reaches_actual_mission_plss_step_and_is_restored() {
        let cmd = command(PoweredGloveMode::Transparent, 0.0);
        let mut glove = PoweredGloveTwin::simulation_reference();
        let step = glove.step(cmd, 1.0).unwrap();

        let clean = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &cmd,
            &step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();
        let mut dusty_modifiers = DurabilityMissionModifiers::clean_reference();
        dusty_modifiers.radiator_heat_rejection_fraction = 0.5;
        let dusty = compose_fidelity_segment(
            &base_segment(),
            dusty_modifiers,
            &cmd,
            &step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();

        let mut clean_mission = IntegratedEvaMission::simulation_reference();
        let mut dusty_mission = IntegratedEvaMission::simulation_reference();
        step_fidelity_segment(&mut clean_mission, &clean).unwrap();
        step_fidelity_segment(&mut dusty_mission, &dusty).unwrap();

        let clean_store = clean_mission
            .plss_mut_for_fault_injection()
            .state()
            .thermal_store_k;
        let dusty_store = dusty_mission
            .plss_mut_for_fault_injection()
            .state()
            .thermal_store_k;
        assert!(dusty_store > clean_store);
        assert_eq!(
            dusty_mission
                .plss_mut_for_fault_injection()
                .environment_heat_rejection_fraction(),
            1.0
        );
    }

    #[test]
    fn isometric_grip_has_explicit_nonzero_equivalent_workload() {
        let cmd = command(PoweredGloveMode::Transparent, 0.0);
        let mut glove = PoweredGloveTwin::simulation_reference();
        let step = glove.step(cmd, 1.0).unwrap();
        let coupled = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &cmd,
            &step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();
        assert_eq!(coupled.dynamic_hand_mechanical_power_w, 0.0);
        assert!(coupled.isometric_hand_equivalent_power_w > 0.0);
    }

    #[test]
    fn relaxed_static_hand_does_not_incur_isometric_workload() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        let cmd = PoweredGloveCommand {
            mode: PoweredGloveMode::Transparent,
            requested_contact_force_n: [0.0; NUM_GLOVE_DIGITS],
            tendon_speed_m_s: [0.0; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
        };
        let step = glove.step(cmd, 1.0).unwrap();
        let coupled = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &cmd,
            &step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();
        assert_eq!(coupled.total_hand_equivalent_power_w, 0.0);
    }

    #[test]
    fn hand_work_increases_actual_integrated_metabolic_estimate() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        let active_cmd = command(PoweredGloveMode::Transparent, 0.02);
        let active_step = glove.step(active_cmd, 1.0).unwrap();
        let active = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &active_cmd,
            &active_step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();

        let mut idle_glove = PoweredGloveTwin::simulation_reference();
        let idle_cmd = PoweredGloveCommand {
            mode: PoweredGloveMode::Transparent,
            requested_contact_force_n: [0.0; NUM_GLOVE_DIGITS],
            tendon_speed_m_s: [0.0; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
        };
        let idle_step = idle_glove.step(idle_cmd, 1.0).unwrap();
        let idle = compose_fidelity_segment(
            &base_segment(),
            DurabilityMissionModifiers::clean_reference(),
            &idle_cmd,
            &idle_step,
            EvaFidelityCouplingConfig::simulation_reference(),
        )
        .unwrap();

        let mut active_mission = IntegratedEvaMission::simulation_reference();
        let mut idle_mission = IntegratedEvaMission::simulation_reference();
        let active_report = step_fidelity_segment(&mut active_mission, &active).unwrap();
        let idle_report = step_fidelity_segment(&mut idle_mission, &idle).unwrap();
        assert!(
            active_report.metabolism.unwrap().metabolic_power_w
                > idle_report.metabolism.unwrap().metabolic_power_w
        );
    }
}
