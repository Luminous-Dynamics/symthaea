// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic fault-injection plans for runtime containment verification.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::execution_guard::{
    ContainmentAction, ExecutionGuard, ExecutionGuardPolicy, ExecutionTelemetry,
};
use serde::{Deserialize, Serialize};

pub const FAULT_INJECTION_SCHEMA: &str = "symthaea.fabrication.fault-injection-plan.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FaultScenario {
    HeartbeatLoss,
    ProgressStall,
    NozzleRunaway,
    BedRunaway,
    NozzleControlDeviation,
    BedControlDeviation,
    TimeRegression,
    ProgressRegression,
    NonFiniteSensor,
}

impl FaultScenario {
    pub fn expected_action(self) -> ContainmentAction {
        match self {
            Self::HeartbeatLoss | Self::ProgressRegression => ContainmentAction::Cancel,
            Self::ProgressStall | Self::NozzleControlDeviation | Self::BedControlDeviation => {
                ContainmentAction::Pause
            }
            Self::NozzleRunaway
            | Self::BedRunaway
            | Self::TimeRegression
            | Self::NonFiniteSensor => ContainmentAction::EmergencyStop,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultInjectionPlan {
    pub schema_version: String,
    pub scenario: FaultScenario,
    /// Optional explicit trigger time. Zero selects the minimum deterministic
    /// time required to cross the configured containment threshold.
    pub trigger_elapsed_s: f64,
}

impl FaultInjectionPlan {
    pub fn new(scenario: FaultScenario) -> Self {
        Self {
            schema_version: FAULT_INJECTION_SCHEMA.into(),
            scenario,
            trigger_elapsed_s: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultObservation {
    pub telemetry: ExecutionTelemetry,
    pub action: ContainmentAction,
    pub latched_action: ContainmentAction,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaultInjectionReport {
    pub schema_version: String,
    pub scenario: FaultScenario,
    pub expected_action: ContainmentAction,
    pub observations: Vec<FaultObservation>,
    pub terminal_action: ContainmentAction,
    pub passed: bool,
    pub report_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FaultInjectionError {
    UnsupportedSchema,
    InvalidTriggerTime,
    InvalidPolicy(&'static str),
    Encoding(String),
}

pub fn run_fault_injection(
    plan: &FaultInjectionPlan,
    policy: ExecutionGuardPolicy,
) -> Result<FaultInjectionReport, FaultInjectionError> {
    if plan.schema_version != FAULT_INJECTION_SCHEMA {
        return Err(FaultInjectionError::UnsupportedSchema);
    }
    if !plan.trigger_elapsed_s.is_finite() || plan.trigger_elapsed_s < 0.0 {
        return Err(FaultInjectionError::InvalidTriggerTime);
    }
    policy
        .validate()
        .map_err(FaultInjectionError::InvalidPolicy)?;
    let mut guard = ExecutionGuard::new(policy)
        .map_err(|_| FaultInjectionError::InvalidPolicy("execution guard policy"))?;
    let initial = baseline(0.0, 1, 0.5);
    let first = guard.observe(initial);
    let trigger_time = if plan.trigger_elapsed_s > 0.0 {
        plan.trigger_elapsed_s
    } else {
        default_trigger_time(plan.scenario, policy)
    };
    let injected = inject(plan.scenario, trigger_time, policy);
    let second = guard.observe(injected);
    let observations = vec![observation(initial, first), observation(injected, second)];
    let terminal_action = guard.latched_action();
    let expected_action = plan.scenario.expected_action();
    let passed = terminal_action >= expected_action;
    let report_digest = digest_report_fields(
        plan.scenario,
        expected_action,
        &observations,
        terminal_action,
        passed,
    )?;
    Ok(FaultInjectionReport {
        schema_version: "symthaea.fabrication.fault-injection-report.v1".into(),
        scenario: plan.scenario,
        expected_action,
        observations,
        terminal_action,
        passed,
        report_digest,
    })
}

pub fn run_standard_fault_matrix(
    policy: ExecutionGuardPolicy,
) -> Result<Vec<FaultInjectionReport>, FaultInjectionError> {
    [
        FaultScenario::HeartbeatLoss,
        FaultScenario::ProgressStall,
        FaultScenario::NozzleRunaway,
        FaultScenario::BedRunaway,
        FaultScenario::NozzleControlDeviation,
        FaultScenario::BedControlDeviation,
        FaultScenario::TimeRegression,
        FaultScenario::ProgressRegression,
        FaultScenario::NonFiniteSensor,
    ]
    .into_iter()
    .map(|scenario| run_fault_injection(&FaultInjectionPlan::new(scenario), policy))
    .collect()
}

pub fn verify_fault_injection_report(
    report: &FaultInjectionReport,
) -> Result<bool, FaultInjectionError> {
    let expected = digest_report_fields(
        report.scenario,
        report.expected_action,
        &report.observations,
        report.terminal_action,
        report.passed,
    )?;
    Ok(expected == report.report_digest
        && report.expected_action == report.scenario.expected_action()
        && report.passed == (report.terminal_action >= report.expected_action))
}

fn default_trigger_time(scenario: FaultScenario, policy: ExecutionGuardPolicy) -> f64 {
    match scenario {
        FaultScenario::HeartbeatLoss => policy.heartbeat_timeout_s + 1.0,
        FaultScenario::ProgressStall => policy.progress_stall_timeout_s + 1.0,
        FaultScenario::NozzleControlDeviation | FaultScenario::BedControlDeviation => {
            policy.thermal_settle_time_s + 1.0
        }
        FaultScenario::TimeRegression => -1.0,
        _ => 1.0,
    }
}

fn baseline(elapsed_s: f64, heartbeat_sequence: u64, progress: f32) -> ExecutionTelemetry {
    ExecutionTelemetry {
        elapsed_s,
        heartbeat_sequence,
        progress,
        nozzle_actual_c: 200.0,
        nozzle_target_c: 200.0,
        bed_actual_c: 60.0,
        bed_target_c: 60.0,
    }
}

fn inject(
    scenario: FaultScenario,
    elapsed_s: f64,
    policy: ExecutionGuardPolicy,
) -> ExecutionTelemetry {
    let mut sample = baseline(elapsed_s, 2, 0.6);
    match scenario {
        FaultScenario::HeartbeatLoss => {
            sample.heartbeat_sequence = 1;
        }
        FaultScenario::ProgressStall => {
            sample.progress = 0.5;
        }
        FaultScenario::NozzleRunaway => {
            sample.nozzle_actual_c = policy.absolute_max_nozzle_c + 1.0;
        }
        FaultScenario::BedRunaway => {
            sample.bed_actual_c = policy.absolute_max_bed_c + 1.0;
        }
        FaultScenario::NozzleControlDeviation => {
            sample.nozzle_actual_c = sample.nozzle_target_c + policy.max_nozzle_deviation_c + 1.0;
        }
        FaultScenario::BedControlDeviation => {
            sample.bed_actual_c = sample.bed_target_c + policy.max_bed_deviation_c + 1.0;
        }
        FaultScenario::TimeRegression => {
            sample.elapsed_s = -1.0;
        }
        FaultScenario::ProgressRegression => {
            sample.progress = 0.4;
        }
        FaultScenario::NonFiniteSensor => {
            sample.nozzle_actual_c = f32::NAN;
        }
    }
    sample
}

fn observation(
    telemetry: ExecutionTelemetry,
    decision: crate::execution_guard::GuardDecision,
) -> FaultObservation {
    FaultObservation {
        telemetry,
        action: decision.action,
        latched_action: decision.latched_action,
        violations: decision
            .new_violations
            .iter()
            .map(|violation| format!("{violation:?}"))
            .collect(),
    }
}

fn digest_report_fields(
    scenario: FaultScenario,
    expected_action: ContainmentAction,
    observations: &[FaultObservation],
    terminal_action: ContainmentAction,
    passed: bool,
) -> Result<Sha256Digest, FaultInjectionError> {
    #[derive(Serialize)]
    struct Body<'a> {
        scenario: FaultScenario,
        expected_action: ContainmentAction,
        observations: &'a [FaultObservation],
        terminal_action: ContainmentAction,
        passed: bool,
    }
    let bytes = serde_json::to_vec(&Body {
        scenario,
        expected_action,
        observations,
        terminal_action,
        passed,
    })
    .map_err(|error| FaultInjectionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.fault-injection-report-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_matrix_reaches_every_expected_containment_level() {
        let reports = run_standard_fault_matrix(ExecutionGuardPolicy::default()).unwrap();
        assert_eq!(reports.len(), 9);
        assert!(reports.iter().all(|report| report.passed));
        assert!(
            reports
                .iter()
                .all(|report| verify_fault_injection_report(report).unwrap())
        );
    }

    #[test]
    fn report_tampering_is_detected() {
        let mut report = run_fault_injection(
            &FaultInjectionPlan::new(FaultScenario::NozzleRunaway),
            ExecutionGuardPolicy::default(),
        )
        .unwrap();
        report.passed = false;
        assert!(!verify_fault_injection_report(&report).unwrap());
    }
}
