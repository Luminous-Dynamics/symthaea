// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hazard-derived motor safety and command arbitration.
//!
//! Consciousness and moral safety remain authoritative caps, but physical
//! hazards must independently constrain actuation before a command reaches the
//! plant. This module keeps that policy deterministic, inspectable, and easy to
//! regression-test.

use crate::embodiment::MotorSafetyLevel;
use crate::geology::GeologicalLookahead;
use crate::observation_quality::ObservationQualityReport;
use crate::path_memory::ReturnPathAssessment;
use crate::recovery_planner::{RecoveryPlan, RecoveryPlanner, VerifiedRecoveryPlanner};
use crate::simulator::RecoveryResources;
use crate::types::{SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubterraneanHazard {
    None,
    Thermal,
    Flood,
    Gas,
    RoofInstability,
    EscapeLoss,
    LocalizationLoss,
    CommunicationsLoss,
    BatteryCritical,
    SpoilJam,
    ReturnReserve,
    TunnelConflict,
    GeologicalUncertainty,
    SensorFault,
}

impl SubterraneanHazard {
    pub const COUNT: usize = 13;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Thermal,
        Self::Flood,
        Self::Gas,
        Self::RoofInstability,
        Self::EscapeLoss,
        Self::LocalizationLoss,
        Self::CommunicationsLoss,
        Self::BatteryCritical,
        Self::SpoilJam,
        Self::ReturnReserve,
        Self::TunnelConflict,
        Self::GeologicalUncertainty,
        Self::SensorFault,
    ];

    pub const fn index(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Thermal => Some(0),
            Self::Flood => Some(1),
            Self::Gas => Some(2),
            Self::RoofInstability => Some(3),
            Self::EscapeLoss => Some(4),
            Self::LocalizationLoss => Some(5),
            Self::CommunicationsLoss => Some(6),
            Self::BatteryCritical => Some(7),
            Self::SpoilJam => Some(8),
            Self::ReturnReserve => Some(9),
            Self::TunnelConflict => Some(10),
            Self::GeologicalUncertainty => Some(11),
            Self::SensorFault => Some(12),
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Thermal => "thermal",
            Self::Flood => "flood",
            Self::Gas => "gas",
            Self::RoofInstability => "roof_instability",
            Self::EscapeLoss => "escape_loss",
            Self::LocalizationLoss => "localization_loss",
            Self::CommunicationsLoss => "communications_loss",
            Self::BatteryCritical => "battery_critical",
            Self::SpoilJam => "spoil_jam",
            Self::ReturnReserve => "return_reserve",
            Self::TunnelConflict => "tunnel_conflict",
            Self::GeologicalUncertainty => "geological_uncertainty",
            Self::SensorFault => "sensor_fault",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HazardAssessment {
    pub primary: SubterraneanHazard,
    pub safety_level: MotorSafetyLevel,
    pub severity: f32,
}

impl HazardAssessment {
    pub const fn clear() -> Self {
        Self {
            primary: SubterraneanHazard::None,
            safety_level: MotorSafetyLevel::Green,
            severity: 0.0,
        }
    }
}

fn safety_rank(level: MotorSafetyLevel) -> u8 {
    match level {
        MotorSafetyLevel::Green => 0,
        MotorSafetyLevel::Yellow => 1,
        MotorSafetyLevel::Orange => 2,
        MotorSafetyLevel::Red => 3,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HazardPortfolio {
    severities: [f32; SubterraneanHazard::COUNT],
}

impl HazardPortfolio {
    pub const fn clear() -> Self {
        Self {
            severities: [0.0; SubterraneanHazard::COUNT],
        }
    }

    pub fn severity(self, hazard: SubterraneanHazard) -> f32 {
        hazard.index().map_or(0.0, |index| self.severities[index])
    }

    pub fn active_count(self, threshold: f32) -> usize {
        self.severities
            .iter()
            .filter(|severity| **severity >= threshold)
            .count()
    }

    pub fn max_severity(self) -> f32 {
        self.severities.iter().copied().fold(0.0, f32::max)
    }

    pub fn primary_assessment(self) -> HazardAssessment {
        let mut primary = SubterraneanHazard::None;
        let mut severity = 0.0f32;
        for hazard in SubterraneanHazard::ALL {
            let candidate = self.severity(hazard);
            if candidate > severity {
                primary = hazard;
                severity = candidate;
            }
        }
        HazardAssessment {
            primary,
            safety_level: level_for(severity),
            severity,
        }
    }

    fn set(&mut self, hazard: SubterraneanHazard, severity: f32) {
        if let Some(index) = hazard.index() {
            self.severities[index] = severity.clamp(0.0, 1.0);
        }
    }

    pub(crate) fn set_max(&mut self, hazard: SubterraneanHazard, severity: f32) {
        if let Some(index) = hazard.index() {
            self.severities[index] = self.severities[index].max(severity.clamp(0.0, 1.0));
        }
    }
}

/// Stateful hazard latch with deterministic de-escalation dwell.
///
/// Entering a severe state is immediate. Leaving it requires consecutive
/// lower-risk observations, preventing actuator chatter at threshold edges and
/// preserving recovery authority long enough for the physical response to
/// become effective.
#[derive(Debug, Clone)]
pub struct HazardSupervisor {
    latched: HazardAssessment,
    raw: HazardAssessment,
    raw_portfolio: HazardPortfolio,
    lower_risk_cycles: u32,
    required_lower_risk_cycles: u32,
}

impl HazardSupervisor {
    pub const DEFAULT_CLEAR_DWELL_CYCLES: u32 = 20;

    pub fn new() -> Self {
        Self::with_clear_dwell(Self::DEFAULT_CLEAR_DWELL_CYCLES)
    }

    pub fn with_clear_dwell(required_lower_risk_cycles: u32) -> Self {
        Self {
            latched: HazardAssessment::clear(),
            raw: HazardAssessment::clear(),
            raw_portfolio: HazardPortfolio::clear(),
            lower_risk_cycles: 0,
            required_lower_risk_cycles: required_lower_risk_cycles.max(1),
        }
    }

    pub fn update(&mut self, state: &SubterraneanState) -> HazardAssessment {
        self.update_portfolio(assess_hazard_portfolio(state))
    }

    pub fn update_with_return_path(
        &mut self,
        state: &SubterraneanState,
        return_path: ReturnPathAssessment,
    ) -> HazardAssessment {
        self.update_portfolio(assess_hazard_portfolio_with_return_path(state, return_path))
    }

    pub fn update_with_operational_context(
        &mut self,
        state: &SubterraneanState,
        return_path: ReturnPathAssessment,
        geology: GeologicalLookahead,
        observation_quality: ObservationQualityReport,
    ) -> HazardAssessment {
        self.update_portfolio(assess_hazard_portfolio_with_operational_context(
            state,
            return_path,
            geology,
            observation_quality,
        ))
    }

    pub(crate) fn update_portfolio(&mut self, portfolio: HazardPortfolio) -> HazardAssessment {
        self.raw_portfolio = portfolio;
        self.raw = self.raw_portfolio.primary_assessment();
        let raw_rank = safety_rank(self.raw.safety_level);
        let latched_rank = safety_rank(self.latched.safety_level);

        let escalate = raw_rank > latched_rank
            || (raw_rank == latched_rank
                && self.raw.severity > self.latched.severity
                && self.raw.primary != SubterraneanHazard::None);
        if escalate {
            self.latched = self.raw;
            self.lower_risk_cycles = 0;
            return self.latched;
        }

        let remains_same_hazard = self.raw.primary == self.latched.primary
            && self.raw.primary != SubterraneanHazard::None
            && raw_rank == latched_rank;
        if remains_same_hazard {
            self.latched.severity = self.raw.severity;
            self.lower_risk_cycles = 0;
            return self.latched;
        }

        if raw_rank < latched_rank || self.raw.primary != self.latched.primary {
            self.lower_risk_cycles = self.lower_risk_cycles.saturating_add(1);
            if self.lower_risk_cycles >= self.required_lower_risk_cycles {
                self.latched = self.raw;
                self.lower_risk_cycles = 0;
            }
        }
        self.latched
    }

    pub fn latched(&self) -> HazardAssessment {
        self.latched
    }

    pub fn raw(&self) -> HazardAssessment {
        self.raw
    }

    pub fn raw_portfolio(&self) -> HazardPortfolio {
        self.raw_portfolio
    }

    pub fn lower_risk_cycles(&self) -> u32 {
        self.lower_risk_cycles
    }

    pub fn reset(&mut self) {
        self.latched = HazardAssessment::clear();
        self.raw = HazardAssessment::clear();
        self.raw_portfolio = HazardPortfolio::clear();
        self.lower_risk_cycles = 0;
    }
}

impl Default for HazardSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

fn normalized_high(value: f64, warning: f64, critical: f64) -> f32 {
    if value <= warning {
        0.0
    } else {
        ((value - warning) / (critical - warning).max(f64::EPSILON)).clamp(0.0, 1.0) as f32
    }
}

fn normalized_low(value: f64, warning: f64, critical: f64) -> f32 {
    if value >= warning {
        0.0
    } else {
        ((warning - value) / (warning - critical).max(f64::EPSILON)).clamp(0.0, 1.0) as f32
    }
}

fn level_for(severity: f32) -> MotorSafetyLevel {
    if severity >= 0.9 {
        MotorSafetyLevel::Red
    } else if severity >= 0.55 {
        MotorSafetyLevel::Orange
    } else if severity > 0.0 {
        MotorSafetyLevel::Yellow
    } else {
        MotorSafetyLevel::Green
    }
}

/// Evaluate physical risk independently of consciousness and moral policy.
///
/// The highest-severity condition wins. Thresholds are deliberately
/// conservative because an enclosed platform cannot assume an immediate human
/// recovery path.
pub fn assess_hazard_portfolio(state: &SubterraneanState) -> HazardPortfolio {
    let mut portfolio = HazardPortfolio::clear();
    if !state.integrity_report().is_valid() {
        portfolio.set(SubterraneanHazard::SensorFault, 1.0);
        return portfolio;
    }

    portfolio.set(
        SubterraneanHazard::Thermal,
        normalized_high(state.cutter_temp_c(), 105.0, 155.0),
    );
    portfolio.set(
        SubterraneanHazard::Flood,
        normalized_high(state.water_ingress_ratio(), 0.25, 0.75)
            .max(normalized_high(state.aquifer_risk(), 0.55, 0.92))
            .max(normalized_low(state.seal_integrity(), 0.65, 0.15)),
    );
    portfolio.set(
        SubterraneanHazard::Gas,
        normalized_high(state.gas_risk(), 0.35, 0.82),
    );
    portfolio.set(
        SubterraneanHazard::RoofInstability,
        normalized_low(state.roof_stability(), 0.58, 0.18),
    );
    portfolio.set(
        SubterraneanHazard::EscapeLoss,
        normalized_low(state.escape_confidence(), 0.55, 0.12).max(normalized_high(
            state.abort_recommendation(),
            0.45,
            0.92,
        )),
    );
    portfolio.set(
        SubterraneanHazard::LocalizationLoss,
        normalized_low(state.localization_confidence(), 0.55, 0.2),
    );
    portfolio.set(
        SubterraneanHazard::CommunicationsLoss,
        normalized_low(state.relay_link_quality(), 0.45, 0.12).max(normalized_low(
            state.comm_signal(),
            0.35,
            0.08,
        )),
    );
    portfolio.set(
        SubterraneanHazard::BatteryCritical,
        normalized_low(state.battery_ratio(), 0.28, 0.05),
    );
    portfolio.set(
        SubterraneanHazard::SpoilJam,
        normalized_high(state.spoil_buffer_fill(), 0.72, 0.96).max(normalized_high(
            state.slurry_load(),
            0.65,
            0.95,
        )),
    );
    portfolio
}

pub fn assess_hazard_portfolio_with_return_path(
    state: &SubterraneanState,
    return_path: ReturnPathAssessment,
) -> HazardPortfolio {
    let mut portfolio = assess_hazard_portfolio(state);
    if portfolio.severity(SubterraneanHazard::SensorFault) > 0.0 {
        return portfolio;
    }
    let margin_severity = normalized_low(return_path.battery_margin, 0.16, 0.0);
    let confidence_severity = normalized_low(return_path.path_confidence, 0.55, 0.2);
    let feasibility_severity = if return_path.feasible { 0.0 } else { 0.9 };
    portfolio.set(
        SubterraneanHazard::ReturnReserve,
        margin_severity
            .max(confidence_severity)
            .max(feasibility_severity),
    );
    portfolio
}

pub fn assess_hazard_portfolio_with_operational_context(
    state: &SubterraneanState,
    return_path: ReturnPathAssessment,
    geology: GeologicalLookahead,
    observation_quality: ObservationQualityReport,
) -> HazardPortfolio {
    let mut portfolio = assess_hazard_portfolio_with_return_path(state, return_path);
    if portfolio.severity(SubterraneanHazard::SensorFault) > 0.0 {
        return portfolio;
    }
    let risk_severity = normalized_high(geology.risk_score, 0.42, 0.82);
    let confidence_severity = normalized_low(geology.minimum_survey_confidence, 0.75, 0.28);
    let transition_severity = if geology.transition_count > 0 && geology.probe_required {
        0.45
    } else {
        0.0
    };
    let geological_severity = if geology.probe_required {
        risk_severity
            .max(confidence_severity)
            .max(transition_severity)
    } else {
        0.0
    };
    portfolio.set(
        SubterraneanHazard::GeologicalUncertainty,
        geological_severity,
    );
    let precision_severity = normalized_low(observation_quality.aggregate_precision, 0.7, 0.3);
    let critical_severity = if observation_quality.critical_degraded_channels > 0 {
        0.9
    } else {
        0.0
    };
    portfolio.set_max(
        SubterraneanHazard::SensorFault,
        precision_severity.max(critical_severity),
    );
    portfolio
}

pub fn assess_hazards(state: &SubterraneanState) -> HazardAssessment {
    assess_hazard_portfolio(state).primary_assessment()
}

pub fn assess_hazards_with_return_path(
    state: &SubterraneanState,
    return_path: ReturnPathAssessment,
) -> HazardAssessment {
    assess_hazard_portfolio_with_return_path(state, return_path).primary_assessment()
}

/// Constrain a learned command to the physical safety envelope.
///
/// Yellow preserves limited autonomy. Orange replaces dangerous degrees of
/// freedom with a hazard-specific withdrawal or hold. Red uses the same
/// fallback at stronger authority. The thermal pump is only forced on for a
/// thermal condition; it is not treated as a generic emergency actuator.
pub fn plan_command(
    command: SubterraneanCommand,
    state: &SubterraneanState,
    assessment: HazardAssessment,
    effective_level: MotorSafetyLevel,
) -> RecoveryPlan {
    VerifiedRecoveryPlanner.plan(command, state, assessment, effective_level, None)
}

/// Plan with explicit finite recovery resources. Runtime and validation paths
/// should prefer this form so exhausted hardware cannot be commanded.
pub fn plan_command_with_resources(
    command: SubterraneanCommand,
    state: &SubterraneanState,
    assessment: HazardAssessment,
    effective_level: MotorSafetyLevel,
    resources: RecoveryResources,
) -> RecoveryPlan {
    VerifiedRecoveryPlanner.plan(command, state, assessment, effective_level, Some(resources))
}

/// Compound-aware planning path used by runtime and held-out validation.
pub fn plan_command_with_portfolio_resources(
    command: SubterraneanCommand,
    state: &SubterraneanState,
    assessment: HazardAssessment,
    portfolio: HazardPortfolio,
    effective_level: MotorSafetyLevel,
    resources: RecoveryResources,
) -> RecoveryPlan {
    VerifiedRecoveryPlanner.plan_portfolio(
        command,
        state,
        assessment,
        portfolio,
        effective_level,
        Some(resources),
    )
}

/// Backward-compatible command-only wrapper around the verified planner.
pub fn arbitrate_command(
    command: SubterraneanCommand,
    state: &SubterraneanState,
    assessment: HazardAssessment,
    effective_level: MotorSafetyLevel,
) -> SubterraneanCommand {
    plan_command(command, state, assessment, effective_level).command
}

/// Resource-aware command-only wrapper for callers that do not need planner
/// rationale in their telemetry.
pub fn arbitrate_command_with_resources(
    command: SubterraneanCommand,
    state: &SubterraneanState,
    assessment: HazardAssessment,
    effective_level: MotorSafetyLevel,
    resources: RecoveryResources,
) -> SubterraneanCommand {
    plan_command_with_resources(command, state, assessment, effective_level, resources).command
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GAS_RISK, ROOF_STABILITY, WATER_INGRESS_RATIO};

    #[test]
    fn thermal_hazard_forces_cooling_and_stops_boring() {
        let mut state = SubterraneanState::home();
        state.channels[crate::types::CUTTER_TEMP_C] = 170.0;
        let assessment = assess_hazards(&state);
        assert_eq!(assessment.primary, SubterraneanHazard::Thermal);
        assert_eq!(assessment.safety_level, MotorSafetyLevel::Red);

        let mut unsafe_command = SubterraneanCommand::zero();
        unsafe_command.set_cutter_head(1.0);
        let safe = arbitrate_command(unsafe_command, &state, assessment, MotorSafetyLevel::Red);
        assert_eq!(safe.cutter_head(), 0.0);
        assert_eq!(safe.thermal_pump(), 1.0);
    }

    #[test]
    fn gas_hazard_withdraws_without_running_cutter() {
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = 0.95;
        let assessment = assess_hazards(&state);
        let safe = arbitrate_command(
            SubterraneanCommand::zero(),
            &state,
            assessment,
            assessment.safety_level,
        );
        assert!(safe.left_track() < 0.0);
        assert_eq!(safe.cutter_head(), 0.0);
        assert_eq!(safe.recovery.dewatering_pump, 0.0);
    }

    #[test]
    fn flood_hazard_uses_explicit_dewatering_and_sealant() {
        let mut state = SubterraneanState::home();
        state.channels[WATER_INGRESS_RATIO] = 0.9;
        state.channels[crate::types::SEAL_INTEGRITY] = 0.3;
        let assessment = assess_hazards(&state);
        let safe = arbitrate_command(
            SubterraneanCommand::zero(),
            &state,
            assessment,
            assessment.safety_level,
        );
        assert!(safe.recovery.dewatering_pump > 0.0);
        assert!(safe.recovery.sealant_injector > 0.0);
        assert_eq!(safe.cutter_head(), 0.0);
    }

    #[test]
    fn communications_loss_spends_relay_authority_not_track_motion() {
        let mut state = SubterraneanState::home();
        state.channels[crate::types::COMM_SIGNAL] = 0.0;
        state.channels[crate::types::RELAY_LINK_QUALITY] = 0.0;
        let assessment = assess_hazards(&state);
        let safe = arbitrate_command(
            SubterraneanCommand::zero(),
            &state,
            assessment,
            assessment.safety_level,
        );
        assert_eq!(safe.left_track(), 0.0);
        assert_eq!(safe.right_track(), 0.0);
        assert_eq!(safe.recovery.relay_deployer, 1.0);
    }

    #[test]
    fn highest_severity_hazard_wins() {
        let mut state = SubterraneanState::home();
        state.channels[WATER_INGRESS_RATIO] = 0.4;
        state.channels[ROOF_STABILITY] = 0.1;
        let assessment = assess_hazards(&state);
        assert_eq!(assessment.primary, SubterraneanHazard::RoofInstability);
        assert_eq!(assessment.safety_level, MotorSafetyLevel::Red);
    }
    #[test]
    fn severe_hazard_is_latched_until_clear_dwell_completes() {
        let mut supervisor = HazardSupervisor::with_clear_dwell(3);
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = 0.95;
        assert_eq!(
            supervisor.update(&state).safety_level,
            MotorSafetyLevel::Red
        );

        state.channels[GAS_RISK] = 0.0;
        assert_eq!(
            supervisor.update(&state).safety_level,
            MotorSafetyLevel::Red
        );
        assert_eq!(
            supervisor.update(&state).safety_level,
            MotorSafetyLevel::Red
        );
        assert_eq!(
            supervisor.update(&state).safety_level,
            MotorSafetyLevel::Green
        );
    }

    #[test]
    fn more_severe_new_hazard_preempts_existing_latch() {
        let mut supervisor = HazardSupervisor::with_clear_dwell(10);
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = 0.5;
        assert_eq!(supervisor.update(&state).primary, SubterraneanHazard::Gas);

        state.channels[crate::types::CUTTER_TEMP_C] = 180.0;
        let assessment = supervisor.update(&state);
        assert_eq!(assessment.primary, SubterraneanHazard::Thermal);
        assert_eq!(assessment.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn malformed_sensor_state_fails_closed() {
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = f64::NAN;
        let assessment = assess_hazards(&state);
        assert_eq!(assessment.primary, SubterraneanHazard::SensorFault);
        assert_eq!(assessment.safety_level, MotorSafetyLevel::Red);
        let command = arbitrate_command(
            SubterraneanCommand::zero(),
            &state,
            assessment,
            assessment.safety_level,
        );
        assert_eq!(command.left_track(), 0.0);
        assert_eq!(command.cutter_head(), 0.0);
        assert_eq!(command.thermal_pump(), 0.5);
    }

    #[test]
    fn insufficient_return_reserve_is_an_independent_hazard() {
        use crate::path_memory::ReturnPathAssessment;

        let state = SubterraneanState::home();
        let assessment = assess_hazards_with_return_path(
            &state,
            ReturnPathAssessment {
                distance_home_m: 80.0,
                recorded_distance_m: 80.0,
                coverage_ratio: 1.0,
                path_confidence: 0.7,
                obstruction_risk: 0.2,
                estimated_battery_required: 0.4,
                battery_margin: -0.1,
                feasible: false,
            },
        );
        assert_eq!(assessment.primary, SubterraneanHazard::ReturnReserve);
        assert_eq!(assessment.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn uncertain_high_consequence_lookahead_requests_geological_caution() {
        use crate::geology::GeologicalLookahead;
        use crate::observation_quality::ObservationQualityReport;
        use crate::path_memory::ReturnPathAssessment;

        let state = SubterraneanState::home();
        let portfolio = assess_hazard_portfolio_with_operational_context(
            &state,
            ReturnPathAssessment::surface(),
            GeologicalLookahead {
                start_depth_m: 10.0,
                horizon_m: 5.0,
                sampled_strata: 2,
                transition_count: 1,
                max_hardness: 0.4,
                max_permeability: 0.9,
                max_gas_potential: 0.7,
                minimum_roof_cohesion: 0.2,
                minimum_survey_confidence: 0.35,
                risk_score: 0.85,
                probe_required: true,
            },
            ObservationQualityReport::nominal(),
        );
        assert!(portfolio.severity(SubterraneanHazard::GeologicalUncertainty) >= 0.9);
    }

    #[test]
    fn critical_model_residual_degradation_escalates_sensor_fault() {
        use crate::geology::GeologicalLookahead;
        use crate::observation_quality::ObservationQualityReport;
        use crate::path_memory::ReturnPathAssessment;

        let portfolio = assess_hazard_portfolio_with_operational_context(
            &SubterraneanState::home(),
            ReturnPathAssessment::surface(),
            GeologicalLookahead::clear(0.0, 5.0),
            ObservationQualityReport {
                aggregate_precision: 0.42,
                minimum_reliability: 0.1,
                maximum_residual: 0.8,
                degraded_channels: 3,
                critical_degraded_channels: 1,
            },
        );
        assert!(portfolio.severity(SubterraneanHazard::SensorFault) >= 0.9);
    }
}
