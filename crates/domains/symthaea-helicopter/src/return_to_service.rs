// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Return-to-service verification after maintenance or safety action.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaintenanceTaskCriticality {
    Routine,
    Significant,
    SafetyCritical,
    FlightCritical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenanceTaskStatus {
    Completed,
    Failed,
    Deferred,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceTaskEvidence {
    pub task_id: String,
    pub criticality: MaintenanceTaskCriticality,
    pub status: MaintenanceTaskStatus,
    pub technician_id: String,
    pub completed_at_ms: Option<u64>,
    pub evidence_ids: BTreeSet<String>,
    pub independent_inspector_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentInstallationEvidence {
    pub position_id: String,
    pub removed_serial: Option<String>,
    pub installed_part_number: String,
    pub installed_serial: String,
    pub approved_configuration: bool,
    pub traceability_evidence_id: Option<String>,
    pub life_limit_reset_evidence_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReturnToServiceTestKind {
    Inspection,
    BuiltInTest,
    GroundFunctional,
    RotorRun,
    HardwareInLoop,
    CheckFlight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnToServiceTestEvidence {
    pub test_id: String,
    pub kind: ReturnToServiceTestKind,
    pub passed: bool,
    pub executed_at_ms: u64,
    pub artifact_digest: Option<String>,
    pub maximum_deviation: Option<f64>,
    pub allowed_deviation: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReturnToServiceWorkOrder {
    pub work_order_id: String,
    pub aircraft_id: String,
    pub opened_at_ms: u64,
    pub closed_at_ms: Option<u64>,
    pub triggering_action_ids: BTreeSet<String>,
    pub required_task_ids: BTreeSet<String>,
    pub required_test_kinds: BTreeSet<ReturnToServiceTestKind>,
    pub expected_deployment_digest: String,
    pub observed_deployment_digest: String,
    pub expected_calibration_digest: String,
    pub observed_calibration_digest: String,
    pub release_authority_id: Option<String>,
    pub release_evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReturnToServicePolicy {
    pub require_independent_inspection_at_or_above: MaintenanceTaskCriticality,
    pub require_component_traceability: bool,
    pub require_life_limit_reset_evidence: bool,
    pub maximum_evidence_age_ms: u64,
}

impl Default for ReturnToServicePolicy {
    fn default() -> Self {
        Self {
            require_independent_inspection_at_or_above: MaintenanceTaskCriticality::SafetyCritical,
            require_component_traceability: true,
            require_life_limit_reset_evidence: true,
            maximum_evidence_age_ms: 30 * 24 * 60 * 60 * 1_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnToServiceStatus {
    Released,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReturnToServiceIssue {
    InvalidWorkOrderWindow,
    WorkOrderNotClosed,
    MissingRequiredTask {
        task_id: String,
    },
    DuplicateTask {
        task_id: String,
    },
    FailedTask {
        task_id: String,
    },
    DeferredCriticalTask {
        task_id: String,
    },
    MissingTaskEvidence {
        task_id: String,
    },
    MissingTechnician {
        task_id: String,
    },
    MissingIndependentInspection {
        task_id: String,
    },
    SelfInspection {
        task_id: String,
    },
    UnapprovedComponent {
        position_id: String,
        serial: String,
    },
    MissingComponentTraceability {
        position_id: String,
        serial: String,
    },
    MissingLifeLimitResetEvidence {
        position_id: String,
        serial: String,
    },
    DuplicateInstalledSerial {
        serial: String,
    },
    MissingRequiredTest {
        kind: ReturnToServiceTestKind,
    },
    FailedTest {
        test_id: String,
    },
    TestToleranceExceeded {
        test_id: String,
        observed: f64,
        allowed: f64,
    },
    MissingTestArtifact {
        test_id: String,
    },
    StaleTestEvidence {
        test_id: String,
        age_ms: u64,
    },
    DeploymentDigestMismatch,
    CalibrationDigestMismatch,
    MissingReleaseAuthority,
    MissingReleaseEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnToServiceReport {
    pub status: ReturnToServiceStatus,
    pub work_order_id: String,
    pub completed_tasks: usize,
    pub passed_tests: usize,
    pub installed_components: usize,
    pub issues: Vec<ReturnToServiceIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnToServiceError {
    InvalidPolicy,
    EmptyWorkOrderId,
}

pub struct ReturnToServiceGate {
    policy: ReturnToServicePolicy,
}

impl ReturnToServiceGate {
    pub fn new(policy: ReturnToServicePolicy) -> Result<Self, ReturnToServiceError> {
        if policy.maximum_evidence_age_ms == 0 {
            return Err(ReturnToServiceError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        work_order: &ReturnToServiceWorkOrder,
        tasks: &[MaintenanceTaskEvidence],
        components: &[ComponentInstallationEvidence],
        tests: &[ReturnToServiceTestEvidence],
        now_ms: u64,
    ) -> Result<ReturnToServiceReport, ReturnToServiceError> {
        if work_order.work_order_id.trim().is_empty() {
            return Err(ReturnToServiceError::EmptyWorkOrderId);
        }
        let mut issues = Vec::new();
        match work_order.closed_at_ms {
            Some(closed_at) if closed_at < work_order.opened_at_ms => {
                issues.push(ReturnToServiceIssue::InvalidWorkOrderWindow)
            }
            None => issues.push(ReturnToServiceIssue::WorkOrderNotClosed),
            _ => {}
        }

        let mut tasks_by_id = BTreeMap::<&str, Vec<&MaintenanceTaskEvidence>>::new();
        for task in tasks {
            tasks_by_id
                .entry(task.task_id.as_str())
                .or_default()
                .push(task);
        }
        for task_id in &work_order.required_task_ids {
            match tasks_by_id.get(task_id.as_str()) {
                None => issues.push(ReturnToServiceIssue::MissingRequiredTask {
                    task_id: task_id.clone(),
                }),
                Some(matching) if matching.len() > 1 => {
                    issues.push(ReturnToServiceIssue::DuplicateTask {
                        task_id: task_id.clone(),
                    })
                }
                Some(_) => {}
            }
        }
        let mut completed_tasks = 0usize;
        for task in tasks {
            match task.status {
                MaintenanceTaskStatus::Completed => completed_tasks += 1,
                MaintenanceTaskStatus::Failed => issues.push(ReturnToServiceIssue::FailedTask {
                    task_id: task.task_id.clone(),
                }),
                MaintenanceTaskStatus::Deferred
                    if task.criticality >= MaintenanceTaskCriticality::SafetyCritical =>
                {
                    issues.push(ReturnToServiceIssue::DeferredCriticalTask {
                        task_id: task.task_id.clone(),
                    })
                }
                MaintenanceTaskStatus::Missing => {
                    issues.push(ReturnToServiceIssue::MissingRequiredTask {
                        task_id: task.task_id.clone(),
                    })
                }
                MaintenanceTaskStatus::Deferred => {}
            }
            if task.technician_id.trim().is_empty() {
                issues.push(ReturnToServiceIssue::MissingTechnician {
                    task_id: task.task_id.clone(),
                });
            }
            if task.status == MaintenanceTaskStatus::Completed && task.evidence_ids.is_empty() {
                issues.push(ReturnToServiceIssue::MissingTaskEvidence {
                    task_id: task.task_id.clone(),
                });
            }
            if task.criticality >= self.policy.require_independent_inspection_at_or_above {
                match task.independent_inspector_id.as_deref() {
                    None | Some("") => {
                        issues.push(ReturnToServiceIssue::MissingIndependentInspection {
                            task_id: task.task_id.clone(),
                        })
                    }
                    Some(inspector) if inspector == task.technician_id => {
                        issues.push(ReturnToServiceIssue::SelfInspection {
                            task_id: task.task_id.clone(),
                        })
                    }
                    Some(_) => {}
                }
            }
        }

        let mut serials = BTreeSet::new();
        for component in components {
            if !serials.insert(component.installed_serial.as_str()) {
                issues.push(ReturnToServiceIssue::DuplicateInstalledSerial {
                    serial: component.installed_serial.clone(),
                });
            }
            if !component.approved_configuration {
                issues.push(ReturnToServiceIssue::UnapprovedComponent {
                    position_id: component.position_id.clone(),
                    serial: component.installed_serial.clone(),
                });
            }
            if self.policy.require_component_traceability
                && component
                    .traceability_evidence_id
                    .as_deref()
                    .unwrap_or("")
                    .is_empty()
            {
                issues.push(ReturnToServiceIssue::MissingComponentTraceability {
                    position_id: component.position_id.clone(),
                    serial: component.installed_serial.clone(),
                });
            }
            if self.policy.require_life_limit_reset_evidence
                && component
                    .life_limit_reset_evidence_id
                    .as_deref()
                    .unwrap_or("")
                    .is_empty()
            {
                issues.push(ReturnToServiceIssue::MissingLifeLimitResetEvidence {
                    position_id: component.position_id.clone(),
                    serial: component.installed_serial.clone(),
                });
            }
        }

        let observed_kinds = tests
            .iter()
            .filter(|test| test.passed)
            .map(|test| test.kind)
            .collect::<BTreeSet<_>>();
        for kind in &work_order.required_test_kinds {
            if !observed_kinds.contains(kind) {
                issues.push(ReturnToServiceIssue::MissingRequiredTest { kind: *kind });
            }
        }
        let mut passed_tests = 0usize;
        for test in tests {
            if test.passed {
                passed_tests += 1;
            } else {
                issues.push(ReturnToServiceIssue::FailedTest {
                    test_id: test.test_id.clone(),
                });
            }
            if test.artifact_digest.as_deref().unwrap_or("").is_empty() {
                issues.push(ReturnToServiceIssue::MissingTestArtifact {
                    test_id: test.test_id.clone(),
                });
            }
            let age_ms = now_ms.saturating_sub(test.executed_at_ms);
            if age_ms > self.policy.maximum_evidence_age_ms {
                issues.push(ReturnToServiceIssue::StaleTestEvidence {
                    test_id: test.test_id.clone(),
                    age_ms,
                });
            }
            if let (Some(observed), Some(allowed)) =
                (test.maximum_deviation, test.allowed_deviation)
            {
                if !observed.is_finite() || !allowed.is_finite() || observed > allowed {
                    issues.push(ReturnToServiceIssue::TestToleranceExceeded {
                        test_id: test.test_id.clone(),
                        observed,
                        allowed,
                    });
                }
            }
        }

        if work_order.observed_deployment_digest != work_order.expected_deployment_digest {
            issues.push(ReturnToServiceIssue::DeploymentDigestMismatch);
        }
        if work_order.observed_calibration_digest != work_order.expected_calibration_digest {
            issues.push(ReturnToServiceIssue::CalibrationDigestMismatch);
        }
        if work_order
            .release_authority_id
            .as_deref()
            .unwrap_or("")
            .is_empty()
        {
            issues.push(ReturnToServiceIssue::MissingReleaseAuthority);
        }
        if work_order
            .release_evidence_id
            .as_deref()
            .unwrap_or("")
            .is_empty()
        {
            issues.push(ReturnToServiceIssue::MissingReleaseEvidence);
        }

        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let status = if issues.iter().any(issue_is_rejection) {
            ReturnToServiceStatus::Rejected
        } else if issues.is_empty() {
            ReturnToServiceStatus::Released
        } else {
            ReturnToServiceStatus::Incomplete
        };
        Ok(ReturnToServiceReport {
            status,
            work_order_id: work_order.work_order_id.clone(),
            completed_tasks,
            passed_tests,
            installed_components: components.len(),
            issues,
        })
    }
}

fn issue_is_rejection(issue: &ReturnToServiceIssue) -> bool {
    matches!(
        issue,
        ReturnToServiceIssue::InvalidWorkOrderWindow
            | ReturnToServiceIssue::DuplicateTask { .. }
            | ReturnToServiceIssue::FailedTask { .. }
            | ReturnToServiceIssue::DeferredCriticalTask { .. }
            | ReturnToServiceIssue::SelfInspection { .. }
            | ReturnToServiceIssue::UnapprovedComponent { .. }
            | ReturnToServiceIssue::DuplicateInstalledSerial { .. }
            | ReturnToServiceIssue::FailedTest { .. }
            | ReturnToServiceIssue::TestToleranceExceeded { .. }
            | ReturnToServiceIssue::StaleTestEvidence { .. }
            | ReturnToServiceIssue::DeploymentDigestMismatch
            | ReturnToServiceIssue::CalibrationDigestMismatch
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work_order() -> ReturnToServiceWorkOrder {
        ReturnToServiceWorkOrder {
            work_order_id: "WO-1".into(),
            aircraft_id: "A1".into(),
            opened_at_ms: 0,
            closed_at_ms: Some(100),
            triggering_action_ids: BTreeSet::from(["FSA-1".into()]),
            required_task_ids: BTreeSet::from(["TASK-1".into()]),
            required_test_kinds: BTreeSet::from([ReturnToServiceTestKind::GroundFunctional]),
            expected_deployment_digest: "deploy".into(),
            observed_deployment_digest: "deploy".into(),
            expected_calibration_digest: "cal".into(),
            observed_calibration_digest: "cal".into(),
            release_authority_id: Some("inspector".into()),
            release_evidence_id: Some("release".into()),
        }
    }
    fn task() -> MaintenanceTaskEvidence {
        MaintenanceTaskEvidence {
            task_id: "TASK-1".into(),
            criticality: MaintenanceTaskCriticality::SafetyCritical,
            status: MaintenanceTaskStatus::Completed,
            technician_id: "tech".into(),
            completed_at_ms: Some(50),
            evidence_ids: BTreeSet::from(["photo".into()]),
            independent_inspector_id: Some("inspector".into()),
        }
    }
    fn component() -> ComponentInstallationEvidence {
        ComponentInstallationEvidence {
            position_id: "tail-gearbox".into(),
            removed_serial: Some("old".into()),
            installed_part_number: "PN-1".into(),
            installed_serial: "SN-2".into(),
            approved_configuration: true,
            traceability_evidence_id: Some("trace".into()),
            life_limit_reset_evidence_id: Some("life".into()),
        }
    }
    fn test() -> ReturnToServiceTestEvidence {
        ReturnToServiceTestEvidence {
            test_id: "TEST-1".into(),
            kind: ReturnToServiceTestKind::GroundFunctional,
            passed: true,
            executed_at_ms: 90,
            artifact_digest: Some("sha256:test".into()),
            maximum_deviation: Some(0.1),
            allowed_deviation: Some(0.2),
        }
    }

    #[test]
    fn complete_work_order_releases_aircraft() {
        let gate = ReturnToServiceGate::new(ReturnToServicePolicy::default()).unwrap();
        let report = gate
            .assess(&work_order(), &[task()], &[component()], &[test()], 100)
            .unwrap();
        assert_eq!(report.status, ReturnToServiceStatus::Released);
    }

    #[test]
    fn self_inspection_is_rejected() {
        let mut task = task();
        task.independent_inspector_id = Some("tech".into());
        let gate = ReturnToServiceGate::new(ReturnToServicePolicy::default()).unwrap();
        let report = gate
            .assess(&work_order(), &[task], &[component()], &[test()], 100)
            .unwrap();
        assert_eq!(report.status, ReturnToServiceStatus::Rejected);
    }

    #[test]
    fn missing_release_evidence_is_incomplete() {
        let mut work_order = work_order();
        work_order.release_evidence_id = None;
        let gate = ReturnToServiceGate::new(ReturnToServicePolicy::default()).unwrap();
        let report = gate
            .assess(&work_order, &[task()], &[component()], &[test()], 100)
            .unwrap();
        assert_eq!(report.status, ReturnToServiceStatus::Incomplete);
    }
}
