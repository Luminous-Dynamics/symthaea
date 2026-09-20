// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapter from the deterministic push-recovery matrix into the capability observatory.
//!
//! The adapter preserves the benchmark's native metrics and negative case evidence.
//! A completed benchmark with falls remains `Completed`: capability failure is not
//! conflated with infrastructure/execution failure.

use crate::capability_observatory::{
    BenchmarkIdentityV1, CapabilityDomain, CapabilityMeasurementV1,
    CapabilityObservationError, CapabilitySubjectIdentityV1, ExecutionSubstrate,
    FailureEventV1, HumanoidCapabilityObservationV1, MeasurementProvenance, RunDisposition,
};
use crate::recovery_benchmark::PushRecoveryMatrixResult;

pub const PUSH_RECOVERY_OBSERVATORY_ADAPTER_SCHEMA_V1: &str =
    "symthaea.humanoid.push-recovery-observatory-adapter.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushRecoveryObservationContextV1 {
    pub run_id: String,
    pub subject: CapabilitySubjectIdentityV1,
    /// Exact identity of the protocol parameters + force/direction case selection.
    pub case_set_id: String,
    pub environment_id: String,
    pub authority_profile_id: String,
    pub evidence_profile_id: String,
}

pub fn observe_push_recovery_matrix_v1(
    context: PushRecoveryObservationContextV1,
    matrix: &PushRecoveryMatrixResult,
) -> Result<HumanoidCapabilityObservationV1, CapabilityObservationError> {
    let mut measurements = vec![
        measurement("case_count", matrix.cases.len() as f64, "count"),
        measurement("recovery_rate", matrix.recovery_rate, "ratio"),
        measurement("fall_rate", matrix.fall_rate, "ratio"),
        measurement("worst_uprightness", matrix.worst_uprightness, "ratio"),
        measurement(
            "worst_capture_margin_m",
            matrix.worst_capture_margin_m,
            "m",
        ),
        measurement(
            "directional_asymmetry",
            matrix.directional_asymmetry,
            "ratio",
        ),
    ];
    if let Some(mean_recovery_time_s) = matrix.mean_recovery_time_s {
        measurements.push(measurement(
            "mean_recovery_time_s",
            mean_recovery_time_s,
            "s",
        ));
    }

    let failures = matrix
        .cases
        .iter()
        .enumerate()
        .filter_map(|(index, case)| {
            if case.result.fell {
                Some(FailureEventV1 {
                    failure_id: format!("push-case-{index}-fall"),
                    category: "fall".into(),
                    description: format!(
                        "push-recovery case {index} fell: direction={:?}, force_n={}",
                        case.direction, case.force_n
                    ),
                })
            } else if !case.result.recovered {
                Some(FailureEventV1 {
                    failure_id: format!("push-case-{index}-not-recovered"),
                    category: "not-recovered".into(),
                    description: format!(
                        "push-recovery case {index} did not recover within protocol window: direction={:?}, force_n={}",
                        case.direction, case.force_n
                    ),
                })
            } else {
                None
            }
        })
        .collect();

    HumanoidCapabilityObservationV1::new(HumanoidCapabilityObservationV1 {
        run_id: context.run_id,
        domain: CapabilityDomain::BalanceRecovery,
        substrate: ExecutionSubstrate::DeterministicSimulation,
        subject: context.subject,
        benchmark: BenchmarkIdentityV1::Internal {
            protocol_id: "humanoid.push-recovery-matrix".into(),
            protocol_version: "v1".into(),
            case_set_id: context.case_set_id,
        },
        environment_id: context.environment_id,
        authority_profile_id: context.authority_profile_id,
        evidence_profile_id: context.evidence_profile_id,
        disposition: RunDisposition::Completed,
        measurements,
        failures,
    })
}

fn measurement(metric_id: &str, value: f64, unit: &str) -> CapabilityMeasurementV1 {
    CapabilityMeasurementV1 {
        metric_id: metric_id.into(),
        value,
        unit: unit.into(),
        provenance: MeasurementProvenance::Derived,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery_benchmark::{
        PushDirection, PushRecoveryCaseResult, PushRecoveryResult,
    };

    fn subject() -> CapabilitySubjectIdentityV1 {
        CapabilitySubjectIdentityV1 {
            source_head: "subject-head".into(),
            model_or_policy_id: "standing-baseline".into(),
            morphology_id: "humanoid-v1".into(),
            sensor_actuator_profile_id: "simple-sim".into(),
        }
    }

    fn context() -> PushRecoveryObservationContextV1 {
        PushRecoveryObservationContextV1 {
            run_id: "push-run-001".into(),
            subject: subject(),
            case_set_id: "forces-0-180-directions-8".into(),
            environment_id: "simple-humanoid-sim".into(),
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "deterministic-simulation-observation".into(),
        }
    }

    fn matrix(recovered: bool, fell: bool) -> PushRecoveryMatrixResult {
        PushRecoveryMatrixResult {
            cases: vec![PushRecoveryCaseResult {
                direction: PushDirection::Forward,
                force_n: 180.0,
                result: PushRecoveryResult {
                    recovered,
                    fell,
                    recovery_time_s: recovered.then_some(0.5),
                    min_uprightness: if fell { 0.1 } else { 0.95 },
                    min_capture_margin_m: if fell { -0.1 } else { 0.03 },
                    peak_recovery_effort: 0.5,
                    recovery_interventions: 1,
                },
            }],
            recovery_rate: if recovered { 1.0 } else { 0.0 },
            fall_rate: if fell { 1.0 } else { 0.0 },
            worst_uprightness: if fell { 0.1 } else { 0.95 },
            worst_capture_margin_m: if fell { -0.1 } else { 0.03 },
            mean_recovery_time_s: recovered.then_some(0.5),
            directional_asymmetry: 0.0,
        }
    }

    #[test]
    fn successful_matrix_is_preserved_as_completed_observation() {
        let observation = observe_push_recovery_matrix_v1(context(), &matrix(true, false)).unwrap();
        assert_eq!(observation.domain, CapabilityDomain::BalanceRecovery);
        assert_eq!(observation.disposition, RunDisposition::Completed);
        assert!(observation.failures.is_empty());
        assert!(observation
            .measurements()
            .iter()
            .any(|metric| metric.metric_id == "recovery_rate" && metric.value == 1.0));
    }

    #[test]
    fn capability_failure_does_not_masquerade_as_execution_failure() {
        let observation = observe_push_recovery_matrix_v1(context(), &matrix(false, true)).unwrap();
        assert_eq!(observation.disposition, RunDisposition::Completed);
        assert_eq!(observation.failures.len(), 1);
        assert_eq!(observation.failures[0].category, "fall");
    }

    #[test]
    fn exact_internal_case_set_identity_is_retained() {
        let observation = observe_push_recovery_matrix_v1(context(), &matrix(true, false)).unwrap();
        assert_eq!(
            observation.benchmark,
            BenchmarkIdentityV1::Internal {
                protocol_id: "humanoid.push-recovery-matrix".into(),
                protocol_version: "v1".into(),
                case_set_id: "forces-0-180-directions-8".into(),
            }
        );
    }
}
