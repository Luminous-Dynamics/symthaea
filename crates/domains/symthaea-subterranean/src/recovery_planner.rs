// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verified recovery planning boundary.
//!
//! Learned control owns nominal motion. This module owns the deterministic,
//! reviewable transformation from physical hazard evidence into operational
//! recovery commands. Compound hazards are composed only where actuator use is
//! compatible, and finite recovery resources are explicit planning input.

use crate::embodiment::MotorSafetyLevel;
use crate::safety::{HazardAssessment, HazardPortfolio, SubterraneanHazard};
use crate::simulator::RecoveryResources;
use crate::types::{SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryAction {
    Nominal,
    LimitedAutonomy,
    GeologicalProbe,
    ThermalArrest,
    FloodIsolation,
    GasWithdrawal,
    RoofStabilization,
    SpoilClearing,
    ControlledWithdrawal,
    NavigationRecovery,
    SensorIsolation,
    EnergyConservation,
    ReserveProtectedReturn,
    TunnelYield,
    PolicyStop,
}

impl RecoveryAction {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::LimitedAutonomy => "limited_autonomy",
            Self::GeologicalProbe => "geological_probe",
            Self::ThermalArrest => "thermal_arrest",
            Self::FloodIsolation => "flood_isolation",
            Self::GasWithdrawal => "gas_withdrawal",
            Self::RoofStabilization => "roof_stabilization",
            Self::SpoilClearing => "spoil_clearing",
            Self::ControlledWithdrawal => "controlled_withdrawal",
            Self::NavigationRecovery => "navigation_recovery",
            Self::SensorIsolation => "sensor_isolation",
            Self::EnergyConservation => "energy_conservation",
            Self::ReserveProtectedReturn => "reserve_protected_return",
            Self::TunnelYield => "tunnel_yield",
            Self::PolicyStop => "policy_stop",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HazardCoverage {
    bits: u16,
}

impl HazardCoverage {
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    pub fn insert(&mut self, hazard: SubterraneanHazard) {
        if let Some(index) = hazard.index() {
            self.bits |= 1u16 << index;
        }
    }

    pub const fn contains(self, hazard: SubterraneanHazard) -> bool {
        match hazard.index() {
            Some(index) => self.bits & (1u16 << index) != 0,
            None => false,
        }
    }

    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }

    pub fn labels(self) -> Vec<String> {
        SubterraneanHazard::ALL
            .iter()
            .copied()
            .filter(|hazard| self.contains(*hazard))
            .map(|hazard| hazard.label().to_string())
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecoveryPlan {
    pub action: RecoveryAction,
    pub command: SubterraneanCommand,
    /// True when preferred recovery hardware was unavailable and the planner
    /// selected a degraded, conservative alternative.
    pub resource_limited: bool,
    /// Active hazards for which this command contains an explicit mitigation.
    pub addressed_hazards: HazardCoverage,
}

pub trait RecoveryPlanner {
    fn plan(
        &self,
        nominal: SubterraneanCommand,
        state: &SubterraneanState,
        assessment: HazardAssessment,
        effective_level: MotorSafetyLevel,
        resources: Option<RecoveryResources>,
    ) -> RecoveryPlan;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VerifiedRecoveryPlanner;

impl VerifiedRecoveryPlanner {
    fn has_dewatering(resources: Option<RecoveryResources>) -> bool {
        resources.is_none_or(|resources| resources.dewatering_health > 0.05)
    }

    fn has_sealant(resources: Option<RecoveryResources>) -> bool {
        resources.is_none_or(|resources| resources.sealant_ratio > 1e-6)
    }

    fn has_relay(resources: Option<RecoveryResources>) -> bool {
        resources.is_none_or(|resources| resources.relay_units > 0)
    }

    fn has_roof_support(resources: Option<RecoveryResources>) -> bool {
        resources.is_none_or(|resources| resources.roof_support_units > 0)
    }

    fn primary_plan(
        &self,
        mut nominal: SubterraneanCommand,
        state: &SubterraneanState,
        assessment: HazardAssessment,
        effective_level: MotorSafetyLevel,
        resources: Option<RecoveryResources>,
    ) -> RecoveryPlan {
        nominal.sanitize();
        let mut coverage = HazardCoverage::empty();
        coverage.insert(assessment.primary);
        match effective_level {
            MotorSafetyLevel::Green => RecoveryPlan {
                action: RecoveryAction::Nominal,
                command: nominal,
                resource_limited: false,
                addressed_hazards: HazardCoverage::empty(),
            },
            MotorSafetyLevel::Yellow => {
                nominal.limit_magnitude(0.6);
                nominal.set_cutter_head(nominal.cutter_head().min(0.35));
                nominal.set_auger_feed(nominal.auger_feed().min(0.45));
                let action = if assessment.primary == SubterraneanHazard::TunnelConflict {
                    nominal.set_cutter_head(0.0);
                    nominal.set_auger_feed(0.0);
                    nominal.set_left_track(0.0);
                    nominal.set_right_track(0.0);
                    RecoveryAction::TunnelYield
                } else if assessment.primary == SubterraneanHazard::GeologicalUncertainty {
                    nominal.set_cutter_head(nominal.cutter_head().min(0.15));
                    nominal.set_auger_feed(nominal.auger_feed().min(0.2));
                    nominal.set_left_track(nominal.left_track().clamp(-0.1, 0.1));
                    nominal.set_right_track(nominal.right_track().clamp(-0.1, 0.1));
                    nominal.set_thermal_pump(nominal.thermal_pump().max(0.2));
                    RecoveryAction::GeologicalProbe
                } else if assessment.primary == SubterraneanHazard::ReturnReserve {
                    nominal.set_cutter_head(0.0);
                    nominal.set_auger_feed(0.0);
                    nominal.set_left_track(nominal.left_track().min(-0.2));
                    nominal.set_right_track(nominal.right_track().min(-0.2));
                    RecoveryAction::ReserveProtectedReturn
                } else {
                    RecoveryAction::LimitedAutonomy
                };
                RecoveryPlan {
                    action,
                    command: nominal,
                    resource_limited: false,
                    addressed_hazards: coverage,
                }
            }
            MotorSafetyLevel::Orange | MotorSafetyLevel::Red => {
                let strong = matches!(effective_level, MotorSafetyLevel::Red);
                let reverse = if strong { -0.55 } else { -0.3 };
                let mut fallback = SubterraneanCommand::zero();
                let mut resource_limited = false;
                let action = match assessment.primary {
                    SubterraneanHazard::Thermal => {
                        fallback.set_thermal_pump(if strong { 1.0 } else { 0.8 });
                        RecoveryAction::ThermalArrest
                    }
                    SubterraneanHazard::Flood => {
                        fallback.set_left_track(reverse);
                        fallback.set_right_track(reverse);
                        if Self::has_dewatering(resources) {
                            fallback.recovery.dewatering_pump = if strong { 1.0 } else { 0.75 };
                        } else {
                            resource_limited = true;
                        }
                        if state.seal_integrity() < 0.75 {
                            if Self::has_sealant(resources) {
                                fallback.recovery.sealant_injector = if strong { 1.0 } else { 0.6 };
                            } else {
                                resource_limited = true;
                            }
                        }
                        if state.cutter_temp_c() >= 70.0 {
                            fallback.set_thermal_pump(if strong { 1.0 } else { 0.6 });
                        }
                        RecoveryAction::FloodIsolation
                    }
                    SubterraneanHazard::Gas => {
                        fallback.set_left_track(reverse);
                        fallback.set_right_track(reverse);
                        if state.cutter_temp_c() >= 70.0 {
                            fallback.set_thermal_pump(if strong { 1.0 } else { 0.6 });
                        }
                        RecoveryAction::GasWithdrawal
                    }
                    SubterraneanHazard::EscapeLoss => {
                        fallback.set_left_track(reverse);
                        fallback.set_right_track(reverse);
                        if state.cutter_temp_c() >= 70.0 {
                            fallback.set_thermal_pump(if strong { 1.0 } else { 0.6 });
                        }
                        RecoveryAction::ControlledWithdrawal
                    }
                    SubterraneanHazard::RoofInstability => {
                        if Self::has_roof_support(resources) {
                            fallback.recovery.roof_support = 1.0;
                        } else {
                            resource_limited = true;
                        }
                        fallback.set_left_track(reverse * 0.5);
                        fallback.set_right_track(reverse * 0.5);
                        RecoveryAction::RoofStabilization
                    }
                    SubterraneanHazard::SpoilJam => {
                        fallback.set_auger_feed(if strong { 0.9 } else { 0.65 });
                        if state.slurry_load() >= 0.5 {
                            if Self::has_dewatering(resources) {
                                fallback.recovery.dewatering_pump = 0.7;
                            } else {
                                resource_limited = true;
                            }
                        }
                        fallback.set_left_track(reverse * 0.5);
                        fallback.set_right_track(reverse * 0.5);
                        RecoveryAction::SpoilClearing
                    }
                    SubterraneanHazard::LocalizationLoss
                    | SubterraneanHazard::CommunicationsLoss => {
                        if Self::has_relay(resources) {
                            fallback.recovery.relay_deployer = 1.0;
                        } else {
                            resource_limited = true;
                        }
                        if state.cutter_temp_c() >= 85.0 {
                            fallback.set_thermal_pump(0.5);
                        }
                        RecoveryAction::NavigationRecovery
                    }
                    SubterraneanHazard::TunnelConflict => {
                        fallback.set_thermal_pump(if state.cutter_temp_c() >= 85.0 {
                            0.4
                        } else {
                            0.0
                        });
                        RecoveryAction::TunnelYield
                    }
                    SubterraneanHazard::GeologicalUncertainty => {
                        fallback.set_cutter_head(if strong { 0.05 } else { 0.1 });
                        fallback.set_auger_feed(if strong { 0.08 } else { 0.15 });
                        fallback.set_left_track(0.0);
                        fallback.set_right_track(0.0);
                        fallback.set_thermal_pump(0.25);
                        RecoveryAction::GeologicalProbe
                    }
                    SubterraneanHazard::SensorFault => {
                        fallback.set_thermal_pump(0.5);
                        RecoveryAction::SensorIsolation
                    }
                    SubterraneanHazard::BatteryCritical => {
                        if state.depth_m() <= 20.0 {
                            fallback.set_left_track(if strong { -0.2 } else { -0.1 });
                            fallback.set_right_track(if strong { -0.2 } else { -0.1 });
                        }
                        RecoveryAction::EnergyConservation
                    }
                    SubterraneanHazard::ReturnReserve => {
                        fallback.set_left_track(if strong { -0.5 } else { -0.3 });
                        fallback.set_right_track(if strong { -0.5 } else { -0.3 });
                        if state.cutter_temp_c() >= 85.0 {
                            fallback.set_thermal_pump(0.4);
                        }
                        RecoveryAction::ReserveProtectedReturn
                    }
                    SubterraneanHazard::None => {
                        fallback.set_thermal_pump(if strong { 1.0 } else { 0.5 });
                        RecoveryAction::PolicyStop
                    }
                };
                fallback.sanitize();
                RecoveryPlan {
                    action,
                    command: fallback,
                    resource_limited,
                    addressed_hazards: coverage,
                }
            }
        }
    }

    /// Compose compatible recovery mechanisms for all currently active
    /// hazards. Movement remains determined by the primary hazard; secondary
    /// hazards may add cooling, dewatering, sealing, relay, or roof support.
    pub fn plan_portfolio(
        &self,
        nominal: SubterraneanCommand,
        state: &SubterraneanState,
        assessment: HazardAssessment,
        portfolio: HazardPortfolio,
        effective_level: MotorSafetyLevel,
        resources: Option<RecoveryResources>,
    ) -> RecoveryPlan {
        let mut plan = self.primary_plan(nominal, state, assessment, effective_level, resources);
        if matches!(effective_level, MotorSafetyLevel::Green) {
            return plan;
        }

        let active = |hazard| portfolio.severity(hazard) > 0.0;
        if active(SubterraneanHazard::SensorFault) {
            plan.command.set_cutter_head(0.0);
            plan.command.set_auger_feed(0.0);
            plan.command.set_left_track(0.0);
            plan.command.set_right_track(0.0);
            plan.command.recovery = Default::default();
            plan.command.set_thermal_pump(0.5);
            plan.addressed_hazards
                .insert(SubterraneanHazard::SensorFault);
            return plan;
        }

        if active(SubterraneanHazard::Thermal) {
            plan.command.set_cutter_head(0.0);
            plan.command
                .set_thermal_pump(plan.command.thermal_pump().max(
                    if matches!(effective_level, MotorSafetyLevel::Red) {
                        1.0
                    } else {
                        0.8
                    },
                ));
            plan.addressed_hazards.insert(SubterraneanHazard::Thermal);
        }

        if active(SubterraneanHazard::Gas) {
            plan.command.set_cutter_head(0.0);
            plan.command.set_auger_feed(0.0);
            if plan.command.left_track() >= 0.0 && plan.command.right_track() >= 0.0 {
                let reverse = if matches!(effective_level, MotorSafetyLevel::Red) {
                    -0.55
                } else {
                    -0.3
                };
                plan.command.set_left_track(reverse);
                plan.command.set_right_track(reverse);
            }
            plan.addressed_hazards.insert(SubterraneanHazard::Gas);
        }

        if active(SubterraneanHazard::Flood) {
            if Self::has_dewatering(resources) {
                plan.command.recovery.dewatering_pump =
                    plan.command.recovery.dewatering_pump.max(0.75);
            } else {
                plan.resource_limited = true;
            }
            if state.seal_integrity() < 0.75 {
                if Self::has_sealant(resources) {
                    plan.command.recovery.sealant_injector =
                        plan.command.recovery.sealant_injector.max(0.6);
                } else {
                    plan.resource_limited = true;
                }
            }
            plan.addressed_hazards.insert(SubterraneanHazard::Flood);
        }

        if active(SubterraneanHazard::RoofInstability) {
            if Self::has_roof_support(resources) {
                plan.command.recovery.roof_support = 1.0;
            } else {
                plan.resource_limited = true;
            }
            plan.addressed_hazards
                .insert(SubterraneanHazard::RoofInstability);
        }

        if active(SubterraneanHazard::LocalizationLoss)
            || active(SubterraneanHazard::CommunicationsLoss)
        {
            if Self::has_relay(resources) {
                plan.command.recovery.relay_deployer = 1.0;
            } else {
                plan.resource_limited = true;
            }
            if active(SubterraneanHazard::LocalizationLoss) {
                plan.addressed_hazards
                    .insert(SubterraneanHazard::LocalizationLoss);
            }
            if active(SubterraneanHazard::CommunicationsLoss) {
                plan.addressed_hazards
                    .insert(SubterraneanHazard::CommunicationsLoss);
            }
        }

        if active(SubterraneanHazard::EscapeLoss) {
            plan.command.set_cutter_head(0.0);
            plan.addressed_hazards
                .insert(SubterraneanHazard::EscapeLoss);
        }

        if active(SubterraneanHazard::SpoilJam)
            && !active(SubterraneanHazard::Gas)
            && !active(SubterraneanHazard::SensorFault)
        {
            plan.command
                .set_auger_feed(plan.command.auger_feed().max(0.65));
            plan.addressed_hazards.insert(SubterraneanHazard::SpoilJam);
        }

        if active(SubterraneanHazard::BatteryCritical) {
            plan.command.set_cutter_head(0.0);
            plan.command.set_auger_feed(0.0);
            plan.command.recovery.sealant_injector = 0.0;
            if !active(SubterraneanHazard::Flood) {
                plan.command.recovery.dewatering_pump = 0.0;
            }
            plan.addressed_hazards
                .insert(SubterraneanHazard::BatteryCritical);
        }

        if active(SubterraneanHazard::GeologicalUncertainty)
            && !active(SubterraneanHazard::Gas)
            && !active(SubterraneanHazard::SensorFault)
        {
            plan.command
                .set_cutter_head(plan.command.cutter_head().clamp(0.0, 0.12));
            plan.command
                .set_auger_feed(plan.command.auger_feed().clamp(0.0, 0.18));
            plan.command
                .set_left_track(plan.command.left_track().clamp(-0.1, 0.1));
            plan.command
                .set_right_track(plan.command.right_track().clamp(-0.1, 0.1));
            plan.addressed_hazards
                .insert(SubterraneanHazard::GeologicalUncertainty);
        }

        if active(SubterraneanHazard::TunnelConflict) {
            plan.command.set_cutter_head(0.0);
            plan.command.set_auger_feed(0.0);
            plan.command.set_left_track(0.0);
            plan.command.set_right_track(0.0);
            plan.addressed_hazards
                .insert(SubterraneanHazard::TunnelConflict);
        }

        if active(SubterraneanHazard::ReturnReserve) {
            plan.command.set_cutter_head(0.0);
            plan.command.set_auger_feed(0.0);
            if plan.command.left_track() >= 0.0 && plan.command.right_track() >= 0.0 {
                plan.command.set_left_track(-0.3);
                plan.command.set_right_track(-0.3);
            }
            plan.addressed_hazards
                .insert(SubterraneanHazard::ReturnReserve);
        }

        plan.command.sanitize();
        plan
    }
}

impl RecoveryPlanner for VerifiedRecoveryPlanner {
    fn plan(
        &self,
        nominal: SubterraneanCommand,
        state: &SubterraneanState,
        assessment: HazardAssessment,
        effective_level: MotorSafetyLevel,
        resources: Option<RecoveryResources>,
    ) -> RecoveryPlan {
        self.primary_plan(nominal, state, assessment, effective_level, resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safety::{assess_hazard_portfolio, assess_hazards};
    use crate::types::{
        BATTERY_RATIO, GAS_RISK, LOCALIZATION_CONFIDENCE, RELAY_LINK_QUALITY, SEAL_INTEGRITY,
        WATER_INGRESS_RATIO,
    };

    #[test]
    fn depleted_flood_resources_produce_degraded_withdrawal() {
        let mut state = SubterraneanState::home();
        state.channels[WATER_INGRESS_RATIO] = 0.95;
        state.channels[SEAL_INTEGRITY] = 0.2;
        let hazard = assess_hazards(&state);
        let plan = VerifiedRecoveryPlanner.plan(
            SubterraneanCommand::zero(),
            &state,
            hazard,
            MotorSafetyLevel::Red,
            Some(RecoveryResources {
                sealant_ratio: 0.0,
                relay_units: 0,
                roof_support_units: 0,
                dewatering_health: 0.0,
            }),
        );
        assert_eq!(plan.action, RecoveryAction::FloodIsolation);
        assert!(plan.resource_limited);
        assert_eq!(plan.command.recovery.dewatering_pump, 0.0);
        assert_eq!(plan.command.recovery.sealant_injector, 0.0);
        assert!(plan.command.left_track() < 0.0);
    }

    #[test]
    fn exhausted_relay_inventory_is_not_commanded() {
        let mut state = SubterraneanState::home();
        state.channels[RELAY_LINK_QUALITY] = 0.0;
        let hazard = assess_hazards(&state);
        let plan = VerifiedRecoveryPlanner.plan(
            SubterraneanCommand::zero(),
            &state,
            hazard,
            MotorSafetyLevel::Red,
            Some(RecoveryResources {
                sealant_ratio: 1.0,
                relay_units: 0,
                roof_support_units: 3,
                dewatering_health: 1.0,
            }),
        );
        assert!(plan.resource_limited);
        assert_eq!(plan.command.recovery.relay_deployer, 0.0);
    }

    #[test]
    fn flood_blackout_plan_composes_dewatering_sealant_and_relay() {
        let mut state = SubterraneanState::home();
        state.channels[WATER_INGRESS_RATIO] = 0.9;
        state.channels[SEAL_INTEGRITY] = 0.25;
        state.channels[RELAY_LINK_QUALITY] = 0.0;
        state.channels[LOCALIZATION_CONFIDENCE] = 0.1;
        let portfolio = assess_hazard_portfolio(&state);
        let primary = portfolio.primary_assessment();
        let plan = VerifiedRecoveryPlanner.plan_portfolio(
            SubterraneanCommand::zero(),
            &state,
            primary,
            portfolio,
            MotorSafetyLevel::Red,
            Some(RecoveryResources::full()),
        );
        assert!(plan.command.recovery.dewatering_pump > 0.0);
        assert!(plan.command.recovery.sealant_injector > 0.0);
        assert_eq!(plan.command.recovery.relay_deployer, 1.0);
        assert!(plan.addressed_hazards.contains(SubterraneanHazard::Flood));
        assert!(
            plan.addressed_hazards
                .contains(SubterraneanHazard::CommunicationsLoss)
        );
    }

    #[test]
    fn low_battery_gas_withdrawal_disables_nonessential_auger() {
        let mut state = SubterraneanState::home();
        state.channels[GAS_RISK] = 0.95;
        state.channels[BATTERY_RATIO] = 0.04;
        let portfolio = assess_hazard_portfolio(&state);
        let plan = VerifiedRecoveryPlanner.plan_portfolio(
            SubterraneanCommand::zero(),
            &state,
            portfolio.primary_assessment(),
            portfolio,
            MotorSafetyLevel::Red,
            Some(RecoveryResources::full()),
        );
        assert_eq!(plan.command.cutter_head(), 0.0);
        assert_eq!(plan.command.auger_feed(), 0.0);
        assert!(plan.command.left_track() <= 0.0);
        assert!(
            plan.addressed_hazards
                .contains(SubterraneanHazard::BatteryCritical)
        );
    }

    #[test]
    fn return_reserve_hazard_stops_cutting_and_withdraws() {
        let state = SubterraneanState::home();
        let assessment = HazardAssessment {
            primary: SubterraneanHazard::ReturnReserve,
            safety_level: MotorSafetyLevel::Red,
            severity: 1.0,
        };
        let plan = VerifiedRecoveryPlanner.plan(
            SubterraneanCommand::zero(),
            &state,
            assessment,
            MotorSafetyLevel::Red,
            Some(RecoveryResources::full()),
        );
        assert_eq!(plan.action, RecoveryAction::ReserveProtectedReturn);
        assert_eq!(plan.command.cutter_head(), 0.0);
        assert!(plan.command.left_track() < 0.0);
    }

    #[test]
    fn geological_uncertainty_uses_low_energy_probe_command() {
        let state = SubterraneanState::home();
        let assessment = HazardAssessment {
            primary: SubterraneanHazard::GeologicalUncertainty,
            safety_level: MotorSafetyLevel::Orange,
            severity: 0.8,
        };
        let plan = VerifiedRecoveryPlanner.plan(
            SubterraneanCommand::zero(),
            &state,
            assessment,
            MotorSafetyLevel::Orange,
            Some(RecoveryResources::full()),
        );
        assert_eq!(plan.action, RecoveryAction::GeologicalProbe);
        assert!(plan.command.cutter_head() <= 0.1);
        assert_eq!(plan.command.left_track(), 0.0);
    }
}
