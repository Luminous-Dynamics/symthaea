// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic operational acceptance contracts for Campaign IV.
//!
//! These contracts validate interactions between geology, look-ahead probing,
//! return reserve protection, residual-based sensor reliability, mission
//! overrides, and verified command planning. They are intentionally independent
//! of learned-controller quality so a safety regression cannot be hidden by a
//! favorable checkpoint.

use crate::embodiment::MotorSafetyLevel;
use crate::geology::{GeologicalLookahead, GeotechnicalProfile, MaterialClass};
use crate::mission::{MissionManager, SubterraneanMissionIntent};
use crate::observation_quality::{ChannelReliabilityMonitor, ObservationQualityReport};
use crate::path_memory::ReturnPathAssessment;
use crate::recovery_planner::{RecoveryAction, RecoveryPlanner, VerifiedRecoveryPlanner};
use crate::safety::{SubterraneanHazard, assess_hazard_portfolio_with_operational_context};
use crate::simulator::{
    RecoveryResources, SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator,
};
use crate::types::{BATTERY_RATIO, SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalContract {
    GeologyAffectsPlant,
    ProbeBeforeUncertainBoundary,
    ProtectReturnReserve,
    DetectCriticalSensorBias,
}

impl OperationalContract {
    pub const fn label(self) -> &'static str {
        match self {
            Self::GeologyAffectsPlant => "geology_affects_plant",
            Self::ProbeBeforeUncertainBoundary => "probe_before_uncertain_boundary",
            Self::ProtectReturnReserve => "protect_return_reserve",
            Self::DetectCriticalSensorBias => "detect_critical_sensor_bias",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalGateFailure {
    GeologyDidNotChangePenetration,
    GeologyDidNotChangeThermalLoad,
    LookaheadDidNotRequestProbe,
    GeologicalHazardMissing,
    ProbeMissionMissing,
    ProbeCommandTooAggressive,
    ReturnReserveHazardMissing,
    ReturnMissionMissing,
    ReturnCommandStillCuts,
    ReturnCommandDoesNotWithdraw,
    CriticalSensorBiasNotDetected,
    SensorFaultHazardMissing,
    SensorIsolationMissing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalValidationReport {
    pub contract: OperationalContract,
    pub passed: bool,
    pub failures: Vec<OperationalGateFailure>,
    pub metrics: BTreeMap<String, f64>,
}

impl OperationalValidationReport {
    fn new(contract: OperationalContract) -> Self {
        Self {
            contract,
            passed: true,
            failures: Vec::new(),
            metrics: BTreeMap::new(),
        }
    }

    fn fail(&mut self, failure: OperationalGateFailure) {
        self.passed = false;
        self.failures.push(failure);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalValidationSuite {
    pub reports: Vec<OperationalValidationReport>,
    pub passed: bool,
}

impl OperationalValidationSuite {
    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OperationalValidator;

impl OperationalValidator {
    pub fn run_reference_suite(&self) -> OperationalValidationSuite {
        let reports = vec![
            self.validate_geology_affects_plant(),
            self.validate_probe_before_boundary(),
            self.validate_return_reserve(),
            self.validate_sensor_bias_detection(),
        ];
        let passed = reports.iter().all(|report| report.passed);
        OperationalValidationSuite { reports, passed }
    }

    fn validate_geology_affects_plant(&self) -> OperationalValidationReport {
        let mut report = OperationalValidationReport::new(OperationalContract::GeologyAffectsPlant);
        let mut granite = SimpleSubterraneanSimulator::with_geology(
            GeotechnicalProfile::homogeneous(MaterialClass::Granite),
        );
        let mut clay = SimpleSubterraneanSimulator::with_geology(GeotechnicalProfile::homogeneous(
            MaterialClass::Clay,
        ));
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(0.75);
        command.set_auger_feed(0.45);
        for _ in 0..800 {
            granite.step(&command, 0.005);
            clay.step(&command, 0.005);
        }
        let granite_depth = granite.state().depth_m();
        let clay_depth = clay.state().depth_m();
        let granite_temperature = granite.state().cutter_temp_c();
        let clay_temperature = clay.state().cutter_temp_c();
        report
            .metrics
            .insert("granite_depth_m".to_string(), granite_depth);
        report
            .metrics
            .insert("clay_depth_m".to_string(), clay_depth);
        report
            .metrics
            .insert("granite_cutter_temp_c".to_string(), granite_temperature);
        report
            .metrics
            .insert("clay_cutter_temp_c".to_string(), clay_temperature);
        if granite_depth >= clay_depth {
            report.fail(OperationalGateFailure::GeologyDidNotChangePenetration);
        }
        if granite_temperature <= clay_temperature {
            report.fail(OperationalGateFailure::GeologyDidNotChangeThermalLoad);
        }
        report
    }

    fn validate_probe_before_boundary(&self) -> OperationalValidationReport {
        let mut report =
            OperationalValidationReport::new(OperationalContract::ProbeBeforeUncertainBoundary);
        let state = SubterraneanState::home();
        let lookahead = GeotechnicalProfile::reference().lookahead(82.0, 8.0);
        report
            .metrics
            .insert("lookahead_risk".to_string(), lookahead.risk_score);
        report.metrics.insert(
            "survey_confidence".to_string(),
            lookahead.minimum_survey_confidence,
        );
        if !lookahead.probe_required {
            report.fail(OperationalGateFailure::LookaheadDidNotRequestProbe);
        }
        let hazard = assess_hazard_portfolio_with_operational_context(
            &state,
            ReturnPathAssessment::surface(),
            lookahead,
            ObservationQualityReport::nominal(),
        )
        .primary_assessment();
        if hazard.primary != SubterraneanHazard::GeologicalUncertainty {
            report.fail(OperationalGateFailure::GeologicalHazardMissing);
        }
        let mut missions = MissionManager::new(SubterraneanMissionIntent::Explore);
        let mission = missions.update(&state, hazard);
        if mission != SubterraneanMissionIntent::ProbeAhead {
            report.fail(OperationalGateFailure::ProbeMissionMissing);
        }
        let plan = VerifiedRecoveryPlanner.plan(
            SubterraneanCommand::zero(),
            &state,
            hazard,
            hazard.safety_level,
            Some(RecoveryResources::full()),
        );
        if plan.action != RecoveryAction::GeologicalProbe || plan.command.cutter_head() > 0.15 {
            report.fail(OperationalGateFailure::ProbeCommandTooAggressive);
        }
        report
    }

    fn validate_return_reserve(&self) -> OperationalValidationReport {
        let mut report =
            OperationalValidationReport::new(OperationalContract::ProtectReturnReserve);
        let mut state = SubterraneanState::home();
        state.channels[crate::types::DEPTH_M] = 80.0;
        state.channels[crate::types::BATTERY_RATIO] = 0.28;
        let return_path = ReturnPathAssessment {
            distance_home_m: 80.0,
            recorded_distance_m: 80.0,
            coverage_ratio: 1.0,
            path_confidence: 0.72,
            obstruction_risk: 0.24,
            estimated_battery_required: 0.2,
            battery_margin: 0.08,
            feasible: true,
        };
        let hazard = assess_hazard_portfolio_with_operational_context(
            &state,
            return_path,
            GeologicalLookahead::clear(80.0, 6.0),
            ObservationQualityReport::nominal(),
        )
        .primary_assessment();
        report.metrics.insert(
            "return_battery_margin".to_string(),
            return_path.battery_margin,
        );
        report
            .metrics
            .insert("hazard_severity".to_string(), hazard.severity as f64);
        if hazard.primary != SubterraneanHazard::ReturnReserve {
            report.fail(OperationalGateFailure::ReturnReserveHazardMissing);
        }
        let mut missions = MissionManager::new(SubterraneanMissionIntent::FollowVein);
        if missions.update(&state, hazard) != SubterraneanMissionIntent::ReturnHome {
            report.fail(OperationalGateFailure::ReturnMissionMissing);
        }
        let mut nominal = SubterraneanCommand::zero();
        nominal.set_cutter_head(0.8);
        nominal.set_left_track(0.5);
        nominal.set_right_track(0.5);
        let plan = VerifiedRecoveryPlanner.plan(
            nominal,
            &state,
            hazard,
            hazard.safety_level.max(MotorSafetyLevel::Yellow),
            Some(RecoveryResources::full()),
        );
        if plan.command.cutter_head() > 0.0 {
            report.fail(OperationalGateFailure::ReturnCommandStillCuts);
        }
        if plan.command.left_track() >= 0.0 || plan.command.right_track() >= 0.0 {
            report.fail(OperationalGateFailure::ReturnCommandDoesNotWithdraw);
        }
        report
    }

    fn validate_sensor_bias_detection(&self) -> OperationalValidationReport {
        let mut report =
            OperationalValidationReport::new(OperationalContract::DetectCriticalSensorBias);
        let predicted = SubterraneanState::home();
        let mut observed = predicted.clone();
        observed.channels[BATTERY_RATIO] = 0.55;
        let mut monitor = ChannelReliabilityMonitor::new(0.2, 0.55);
        let mut quality = ObservationQualityReport::nominal();
        for _ in 0..30 {
            quality = monitor.update(&predicted, &observed);
        }
        report.metrics.insert(
            "observation_precision".to_string(),
            quality.aggregate_precision,
        );
        report.metrics.insert(
            "critical_degraded_channels".to_string(),
            quality.critical_degraded_channels as f64,
        );
        if !quality.requires_fail_closed() {
            report.fail(OperationalGateFailure::CriticalSensorBiasNotDetected);
        }
        let hazard = assess_hazard_portfolio_with_operational_context(
            &observed,
            ReturnPathAssessment::surface(),
            GeologicalLookahead::clear(0.0, 6.0),
            quality,
        )
        .primary_assessment();
        if hazard.primary != SubterraneanHazard::SensorFault {
            report.fail(OperationalGateFailure::SensorFaultHazardMissing);
        }
        let plan = VerifiedRecoveryPlanner.plan(
            SubterraneanCommand::zero(),
            &observed,
            hazard,
            hazard.safety_level,
            Some(RecoveryResources::full()),
        );
        if plan.action != RecoveryAction::SensorIsolation {
            report.fail(OperationalGateFailure::SensorIsolationMissing);
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_operational_contracts_pass() {
        let suite = OperationalValidator.run_reference_suite();
        assert!(suite.passed, "operational validation failures: {suite:?}");
        assert_eq!(suite.reports.len(), 4);
        assert!(suite.to_pretty_json().is_ok());
    }
}
