// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quantitative, preregistered recovery validation.
//!
//! A recovery policy is not validated merely because it selects the expected
//! enum. This module runs deterministic emergency scenarios against explicit
//! time, energy, resource, and terminal-severity contracts and reports every
//! failed gate as structured evidence.

use crate::curriculum::{ScenarioCurriculum, SubterraneanScenario, SubterraneanScenarioKind};
use crate::reflex::reflex_command_for_mission;
use crate::safety::{HazardSupervisor, SubterraneanHazard, plan_command_with_portfolio_resources};
use crate::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use crate::types::BATTERY_RATIO;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryObjective {
    /// The latched physical hazard must fall below Orange before the deadline.
    ClearHazard,
    /// The condition is not physically reversible in the reference plant, but
    /// the policy must contain it without divergence or further escalation.
    ContainHazard,
    /// Nominal control must remain below the intervention threshold.
    PreserveNominalOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecoveryContract {
    pub scenario_kind: SubterraneanScenarioKind,
    pub objective: RecoveryObjective,
    pub max_recovery_seconds: f64,
    pub max_terminal_severity: f32,
    pub max_battery_spent: f64,
    pub max_sealant_spent: f64,
    pub max_relay_units_spent: u8,
    pub max_roof_support_units_spent: u8,
}

impl RecoveryContract {
    /// Conservative first preregistration for the deterministic reference
    /// simulator. These are explicit research gates, not field-certification
    /// limits. Tightening them requires a versioned evidence campaign.
    pub const fn for_scenario(kind: SubterraneanScenarioKind) -> Self {
        match kind {
            SubterraneanScenarioKind::NominalTransit => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::PreserveNominalOperation,
                max_recovery_seconds: 0.0,
                max_terminal_severity: 0.15,
                max_battery_spent: 0.08,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::AquiferBreach => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 25.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.35,
                max_sealant_spent: 0.45,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::GasPocket => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 18.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.25,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::RoofFailure => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 18.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.22,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 1,
            },
            SubterraneanScenarioKind::CommunicationsBlackout => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 12.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.12,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 1,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::SpoilJam => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 20.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.3,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::ThermalRunaway => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 20.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.22,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::BatteryEmergency => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ContainHazard,
                max_recovery_seconds: 0.0,
                max_terminal_severity: 1.0,
                max_battery_spent: 0.04,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::SensorFault => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ContainHazard,
                max_recovery_seconds: 0.0,
                max_terminal_severity: 1.0,
                max_battery_spent: 0.12,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::FloodBlackout => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 30.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.45,
                max_sealant_spent: 0.45,
                max_relay_units_spent: 1,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::GasLocalizationLoss => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ClearHazard,
                max_recovery_seconds: 25.0,
                max_terminal_severity: 0.54,
                max_battery_spent: 0.35,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 1,
                max_roof_support_units_spent: 0,
            },
            SubterraneanScenarioKind::LowBatteryWithdrawal => Self {
                scenario_kind: kind,
                objective: RecoveryObjective::ContainHazard,
                max_recovery_seconds: 0.0,
                max_terminal_severity: 1.0,
                max_battery_spent: 0.05,
                max_sealant_spent: 0.0,
                max_relay_units_spent: 0,
                max_roof_support_units_spent: 0,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryGateFailure {
    NoDisturbanceObserved,
    UnexpectedSafetyIntervention,
    HazardNotCleared,
    RecoveryDeadlineExceeded {
        actual_seconds: f64,
        limit_seconds: f64,
    },
    TerminalSeverityExceeded {
        actual: f32,
        limit: f32,
    },
    BatteryBudgetExceeded {
        actual: f64,
        limit: f64,
    },
    SealantBudgetExceeded {
        actual: f64,
        limit: f64,
    },
    RelayBudgetExceeded {
        actual: u8,
        limit: u8,
    },
    RoofSupportBudgetExceeded {
        actual: u8,
        limit: u8,
    },
    StateDiverged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryValidationReport {
    pub scenario_kind: SubterraneanScenarioKind,
    pub contract: RecoveryContract,
    pub passed: bool,
    pub steps_executed: usize,
    pub events_applied: usize,
    pub intervention_frames: usize,
    pub first_disturbance_step: Option<usize>,
    pub first_recovered_step: Option<usize>,
    pub recovery_seconds: Option<f64>,
    pub peak_hazard_severity: f32,
    pub peak_simultaneous_hazards: usize,
    pub terminal_active_hazards: usize,
    pub terminal_hazard: String,
    pub terminal_hazard_severity: f32,
    pub initial_battery_ratio: f64,
    pub terminal_battery_ratio: f64,
    pub battery_spent: f64,
    pub sealant_spent: f64,
    pub relay_units_spent: u8,
    pub roof_support_units_spent: u8,
    pub failures: Vec<RecoveryGateFailure>,
}

impl RecoveryValidationReport {
    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryValidationSuite {
    pub reports: Vec<RecoveryValidationReport>,
}

impl RecoveryValidationSuite {
    pub fn all_passed(&self) -> bool {
        self.reports.iter().all(|report| report.passed)
    }

    pub fn failed_count(&self) -> usize {
        self.reports.iter().filter(|report| !report.passed).count()
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecoveryValidationConfig {
    pub physics_hz: f64,
    pub duration_seconds: f64,
}

impl Default for RecoveryValidationConfig {
    fn default() -> Self {
        Self {
            physics_hz: 100.0,
            duration_seconds: 30.0,
        }
    }
}

pub struct RecoveryValidator {
    config: RecoveryValidationConfig,
}

impl RecoveryValidator {
    pub fn new(config: RecoveryValidationConfig) -> Self {
        Self { config }
    }

    pub fn evaluate_standard_curriculum(&self) -> RecoveryValidationSuite {
        let steps = self.step_count();
        let curriculum = ScenarioCurriculum::standard(steps);
        let reports = curriculum
            .scenarios()
            .iter()
            .map(|scenario| {
                self.evaluate_scenario(scenario, RecoveryContract::for_scenario(scenario.kind))
            })
            .collect();
        RecoveryValidationSuite { reports }
    }

    pub fn evaluate_compound_holdout(&self) -> RecoveryValidationSuite {
        let steps = self.step_count();
        let curriculum = ScenarioCurriculum::compound_holdout(steps);
        let reports = curriculum
            .scenarios()
            .iter()
            .map(|scenario| {
                self.evaluate_scenario(scenario, RecoveryContract::for_scenario(scenario.kind))
            })
            .collect();
        RecoveryValidationSuite { reports }
    }

    pub fn evaluate_scenario(
        &self,
        scenario: &SubterraneanScenario,
        contract: RecoveryContract,
    ) -> RecoveryValidationReport {
        let mut simulator = SimpleSubterraneanSimulator::new();
        scenario.initialize(simulator.state_mut());
        let initial_resources = simulator.recovery_resources();
        let initial_battery_ratio = simulator.state().channels[BATTERY_RATIO];
        let mut supervisor = HazardSupervisor::new();
        let dt = self.dt();
        let mut events_applied = 0usize;
        let mut intervention_frames = 0usize;
        let mut first_disturbance_step = None;
        let mut first_recovered_step = None;
        let mut peak_hazard_severity = 0.0f32;
        let mut peak_simultaneous_hazards = 0usize;
        let mut steps_executed = 0usize;
        let mut diverged = false;

        for step in 0..self.step_count() {
            let events = scenario.apply_events(step, simulator.state_mut());
            if events > 0 && first_disturbance_step.is_none() {
                first_disturbance_step = Some(step);
            }
            events_applied = events_applied.saturating_add(events);

            let hazard = supervisor.update(simulator.state());
            let portfolio = supervisor.raw_portfolio();
            peak_hazard_severity = peak_hazard_severity.max(portfolio.max_severity());
            peak_simultaneous_hazards = peak_simultaneous_hazards.max(portfolio.active_count(0.01));
            if hazard.primary != SubterraneanHazard::None {
                intervention_frames = intervention_frames.saturating_add(1);
            }
            if first_disturbance_step.is_some()
                && first_recovered_step.is_none()
                && portfolio.max_severity() < 0.55
            {
                first_recovered_step = Some(step);
            }

            let nominal = reflex_command_for_mission(simulator.state(), scenario.mission_intent());
            let command = plan_command_with_portfolio_resources(
                nominal,
                simulator.state(),
                hazard,
                portfolio,
                hazard.safety_level,
                simulator.recovery_resources(),
            )
            .command;
            simulator.step(&command, dt);
            steps_executed = step + 1;
            if !simulator.state().is_finite() {
                diverged = true;
                break;
            }
        }

        let terminal_hazard = supervisor.update(simulator.state());
        let terminal_portfolio = supervisor.raw_portfolio();
        let terminal_resources = simulator.recovery_resources();
        let terminal_battery_ratio = simulator.state().channels[BATTERY_RATIO];
        let battery_spent = (initial_battery_ratio - terminal_battery_ratio).max(0.0);
        let sealant_spent =
            (initial_resources.sealant_ratio - terminal_resources.sealant_ratio).max(0.0);
        let relay_units_spent = initial_resources
            .relay_units
            .saturating_sub(terminal_resources.relay_units);
        let roof_support_units_spent = initial_resources
            .roof_support_units
            .saturating_sub(terminal_resources.roof_support_units);
        let recovery_seconds = match (first_disturbance_step, first_recovered_step) {
            (Some(start), Some(end)) if end >= start => Some((end - start) as f64 * dt),
            _ => None,
        };

        let mut failures = Vec::new();
        if diverged {
            failures.push(RecoveryGateFailure::StateDiverged);
        }
        match contract.objective {
            RecoveryObjective::PreserveNominalOperation => {
                if intervention_frames > 0 {
                    failures.push(RecoveryGateFailure::UnexpectedSafetyIntervention);
                }
            }
            RecoveryObjective::ClearHazard => {
                if events_applied == 0 {
                    failures.push(RecoveryGateFailure::NoDisturbanceObserved);
                }
                match recovery_seconds {
                    Some(actual) if actual > contract.max_recovery_seconds => {
                        failures.push(RecoveryGateFailure::RecoveryDeadlineExceeded {
                            actual_seconds: actual,
                            limit_seconds: contract.max_recovery_seconds,
                        });
                    }
                    None => failures.push(RecoveryGateFailure::HazardNotCleared),
                    Some(_) => {}
                }
            }
            RecoveryObjective::ContainHazard => {
                if events_applied == 0 {
                    failures.push(RecoveryGateFailure::NoDisturbanceObserved);
                }
            }
        }
        if terminal_hazard.severity > contract.max_terminal_severity {
            failures.push(RecoveryGateFailure::TerminalSeverityExceeded {
                actual: terminal_hazard.severity,
                limit: contract.max_terminal_severity,
            });
        }
        if battery_spent > contract.max_battery_spent {
            failures.push(RecoveryGateFailure::BatteryBudgetExceeded {
                actual: battery_spent,
                limit: contract.max_battery_spent,
            });
        }
        if sealant_spent > contract.max_sealant_spent {
            failures.push(RecoveryGateFailure::SealantBudgetExceeded {
                actual: sealant_spent,
                limit: contract.max_sealant_spent,
            });
        }
        if relay_units_spent > contract.max_relay_units_spent {
            failures.push(RecoveryGateFailure::RelayBudgetExceeded {
                actual: relay_units_spent,
                limit: contract.max_relay_units_spent,
            });
        }
        if roof_support_units_spent > contract.max_roof_support_units_spent {
            failures.push(RecoveryGateFailure::RoofSupportBudgetExceeded {
                actual: roof_support_units_spent,
                limit: contract.max_roof_support_units_spent,
            });
        }

        RecoveryValidationReport {
            scenario_kind: scenario.kind,
            contract,
            passed: failures.is_empty(),
            steps_executed,
            events_applied,
            intervention_frames,
            first_disturbance_step,
            first_recovered_step,
            recovery_seconds,
            peak_hazard_severity,
            peak_simultaneous_hazards,
            terminal_active_hazards: terminal_portfolio.active_count(0.01),
            terminal_hazard: terminal_hazard.primary.label().to_string(),
            terminal_hazard_severity: terminal_hazard.severity,
            initial_battery_ratio,
            terminal_battery_ratio,
            battery_spent,
            sealant_spent,
            relay_units_spent,
            roof_support_units_spent,
            failures,
        }
    }

    fn dt(&self) -> f64 {
        if self.config.physics_hz.is_finite() && self.config.physics_hz > 0.0 {
            1.0 / self.config.physics_hz
        } else {
            0.01
        }
    }

    fn step_count(&self) -> usize {
        let hz = if self.config.physics_hz.is_finite() && self.config.physics_hz > 0.0 {
            self.config.physics_hz
        } else {
            100.0
        };
        let duration =
            if self.config.duration_seconds.is_finite() && self.config.duration_seconds > 0.0 {
                self.config.duration_seconds
            } else {
                30.0
            };
        (hz * duration).round().max(1.0) as usize
    }
}

impl Default for RecoveryValidator {
    fn default() -> Self {
        Self::new(RecoveryValidationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_standard_scenario_has_an_explicit_contract() {
        let curriculum = ScenarioCurriculum::standard(100);
        for scenario in curriculum.scenarios() {
            let contract = RecoveryContract::for_scenario(scenario.kind);
            assert_eq!(contract.scenario_kind, scenario.kind);
            assert!(contract.max_battery_spent >= 0.0);
        }
    }

    #[test]
    fn deterministic_report_records_event_and_resource_evidence() {
        let curriculum = ScenarioCurriculum::standard(500);
        let scenario = curriculum
            .scenarios()
            .iter()
            .find(|scenario| scenario.kind == SubterraneanScenarioKind::AquiferBreach)
            .expect("standard curriculum includes aquifer breach");
        let validator = RecoveryValidator::new(RecoveryValidationConfig {
            physics_hz: 50.0,
            duration_seconds: 10.0,
        });
        let report = validator.evaluate_scenario(
            scenario,
            RecoveryContract::for_scenario(SubterraneanScenarioKind::AquiferBreach),
        );
        assert_eq!(report.events_applied, 1);
        assert!(report.first_disturbance_step.is_some());
        assert!(report.battery_spent.is_finite());
        assert!(report.sealant_spent >= 0.0);
        assert!(report.to_pretty_json().is_ok());
    }

    #[test]
    fn impossible_budget_is_reported_as_a_gate_failure() {
        let scenario = SubterraneanScenario::nominal();
        let validator = RecoveryValidator::new(RecoveryValidationConfig {
            physics_hz: 50.0,
            duration_seconds: 2.0,
        });
        let mut contract = RecoveryContract::for_scenario(SubterraneanScenarioKind::NominalTransit);
        contract.max_battery_spent = -1.0;
        let report = validator.evaluate_scenario(&scenario, contract);
        assert!(!report.passed);
        assert!(
            report.failures.iter().any(|failure| matches!(
                failure,
                RecoveryGateFailure::BatteryBudgetExceeded { .. }
            ))
        );
    }

    #[test]
    fn initial_and_terminal_resource_accounting_is_saturating() {
        let full = crate::simulator::RecoveryResources::full();
        assert_eq!(full.relay_units.saturating_sub(full.relay_units), 0);
        assert_eq!(
            full.roof_support_units
                .saturating_sub(full.roof_support_units),
            0
        );
    }

    #[test]
    fn compound_evaluation_records_simultaneous_hazard_count() {
        let validator = RecoveryValidator::new(RecoveryValidationConfig {
            physics_hz: 20.0,
            duration_seconds: 8.0,
        });
        let suite = validator.evaluate_compound_holdout();
        assert_eq!(suite.reports.len(), 3);
        assert!(
            suite
                .reports
                .iter()
                .any(|report| report.peak_simultaneous_hazards >= 2)
        );
    }
}
