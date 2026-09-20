// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Push-recovery integration with the capability observatory.
//!
//! The public runner derives a deterministic case-set commitment from the exact
//! protocol + ordered force levels before executing the matrix, so result identity
//! cannot be supplied as an unrelated free-form label.

use crate::capability_observatory::{
    BenchmarkIdentityV1, CapabilityDomain, CapabilityMeasurementV1,
    CapabilityObservationError, CapabilitySubjectIdentityV1, ExecutionSubstrate,
    FailureEventV1, HumanoidCapabilityObservationV1, MeasurementProvenance, RunDisposition,
};
use crate::recovery_benchmark::{
    PushRecoveryMatrixResult, PushRecoveryProtocol, run_push_recovery_matrix,
};
use crate::simulator::HumanoidPhysicsSimulator;

pub const PUSH_RECOVERY_OBSERVATORY_ADAPTER_SCHEMA_V1: &str =
    "symthaea.humanoid.push-recovery-observatory-adapter.v1";
pub const PUSH_RECOVERY_CASE_SET_SCHEMA_V1: &str =
    "symthaea.humanoid.push-recovery-case-set.v1";

const PUSH_RECOVERY_CASE_SET_DOMAIN_V1: &[u8] =
    b"symthaea.humanoid.push-recovery-case-set.v1\0";
const PUSH_DIRECTION_SET_V1: &[u8] =
    b"forward,backward,left,right,forward-left,forward-right,backward-left,backward-right\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushRecoveryObservationContextV1 {
    pub run_id: String,
    pub subject: CapabilitySubjectIdentityV1,
    pub environment_id: String,
    pub authority_profile_id: String,
    pub evidence_profile_id: String,
}

/// Deterministically identify the exact push-recovery matrix semantics.
///
/// `PushRecoveryProtocol::push_force_n` is intentionally excluded because
/// `run_push_recovery_matrix` overwrites it for every generated case.
pub fn push_recovery_case_set_id_v1(
    protocol: &PushRecoveryProtocol,
    force_levels_n: &[f64],
) -> Result<String, CapabilityObservationError> {
    if !protocol.physics_hz.is_finite()
        || protocol.physics_hz <= 0.0
        || !protocol.settle_seconds.is_finite()
        || protocol.settle_seconds < 0.0
        || !protocol.evaluate_seconds.is_finite()
        || protocol.evaluate_seconds < 0.0
        || !protocol.recovered_margin_m.is_finite()
        || !protocol.recovered_uprightness.is_finite()
        || !(0.0..=1.0).contains(&protocol.recovered_uprightness)
        || !protocol.recovered_hold_seconds.is_finite()
        || protocol.recovered_hold_seconds < 0.0
        || force_levels_n.is_empty()
        || force_levels_n
            .iter()
            .any(|force| !force.is_finite() || *force < 0.0)
    {
        return Err(CapabilityObservationError::InvalidBenchmarkIdentity);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(PUSH_RECOVERY_CASE_SET_DOMAIN_V1);
    update_f64(&mut hasher, b"physics_hz\0", protocol.physics_hz);
    update_f64(
        &mut hasher,
        b"settle_seconds\0",
        protocol.settle_seconds,
    );
    update_f64(
        &mut hasher,
        b"evaluate_seconds\0",
        protocol.evaluate_seconds,
    );
    update_f64(
        &mut hasher,
        b"recovered_margin_m\0",
        protocol.recovered_margin_m,
    );
    update_f64(
        &mut hasher,
        b"recovered_uprightness\0",
        protocol.recovered_uprightness,
    );
    update_f64(
        &mut hasher,
        b"recovered_hold_seconds\0",
        protocol.recovered_hold_seconds,
    );
    hasher.update(b"direction_set_v1\0");
    hasher.update(PUSH_DIRECTION_SET_V1);
    hasher.update(b"force_count\0");
    hasher.update(&(force_levels_n.len() as u64).to_le_bytes());
    for force in force_levels_n {
        hasher.update(b"force_n\0");
        hasher.update(&force.to_le_bytes());
    }

    Ok(format!("push-recovery-v1:{}", hasher.finalize().to_hex()))
}

/// Execute the exact matrix and emit its observation from the same protocol inputs.
///
/// This combined runner prevents an already-computed matrix from being paired with
/// an unrelated caller-supplied case-set label.
pub fn run_and_observe_push_recovery_matrix_v1(
    simulator: &mut dyn HumanoidPhysicsSimulator,
    context: PushRecoveryObservationContextV1,
    protocol: &PushRecoveryProtocol,
    force_levels_n: &[f64],
) -> Result<(PushRecoveryMatrixResult, HumanoidCapabilityObservationV1), CapabilityObservationError>
{
    let case_set_id = push_recovery_case_set_id_v1(protocol, force_levels_n)?;
    let matrix = run_push_recovery_matrix(simulator, protocol, force_levels_n);
    let observation = observation_from_matrix_v1(context, case_set_id, &matrix)?;
    Ok((matrix, observation))
}

fn observation_from_matrix_v1(
    context: PushRecoveryObservationContextV1,
    case_set_id: String,
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
            case_set_id,
        },
        environment_id: context.environment_id,
        authority_profile_id: context.authority_profile_id,
        evidence_profile_id: context.evidence_profile_id,
        disposition: RunDisposition::Completed,
        measurements,
        failures,
    })
}

fn update_f64(hasher: &mut blake3::Hasher, label: &[u8], value: f64) {
    hasher.update(label);
    hasher.update(&value.to_le_bytes());
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
    fn case_set_commitment_is_deterministic_and_force_sensitive() {
        let protocol = PushRecoveryProtocol::default();
        let a = push_recovery_case_set_id_v1(&protocol, &[0.0, 90.0, 180.0]).unwrap();
        let b = push_recovery_case_set_id_v1(&protocol, &[0.0, 90.0, 180.0]).unwrap();
        let changed = push_recovery_case_set_id_v1(&protocol, &[0.0, 90.0, 181.0]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, changed);
    }

    #[test]
    fn matrix_template_push_force_is_not_part_of_case_set_identity() {
        let a = PushRecoveryProtocol::default();
        let mut b = a.clone();
        b.push_force_n = [999.0, -123.0, 42.0];
        assert_eq!(
            push_recovery_case_set_id_v1(&a, &[180.0]).unwrap(),
            push_recovery_case_set_id_v1(&b, &[180.0]).unwrap()
        );
    }

    #[test]
    fn invalid_force_levels_fail_closed() {
        let protocol = PushRecoveryProtocol::default();
        assert_eq!(
            push_recovery_case_set_id_v1(&protocol, &[]),
            Err(CapabilityObservationError::InvalidBenchmarkIdentity)
        );
        assert_eq!(
            push_recovery_case_set_id_v1(&protocol, &[-1.0]),
            Err(CapabilityObservationError::InvalidBenchmarkIdentity)
        );
    }

    #[test]
    fn successful_matrix_preserves_derived_case_set_identity() {
        let protocol = PushRecoveryProtocol::default();
        let case_set_id = push_recovery_case_set_id_v1(&protocol, &[180.0]).unwrap();
        let observation =
            observation_from_matrix_v1(context(), case_set_id.clone(), &matrix(true, false))
                .unwrap();
        assert_eq!(observation.domain, CapabilityDomain::BalanceRecovery);
        assert_eq!(observation.disposition, RunDisposition::Completed);
        assert!(observation.failures.is_empty());
        assert_eq!(
            observation.benchmark,
            BenchmarkIdentityV1::Internal {
                protocol_id: "humanoid.push-recovery-matrix".into(),
                protocol_version: "v1".into(),
                case_set_id,
            }
        );
    }

    #[test]
    fn capability_failure_does_not_masquerade_as_execution_failure() {
        let protocol = PushRecoveryProtocol::default();
        let case_set_id = push_recovery_case_set_id_v1(&protocol, &[180.0]).unwrap();
        let observation =
            observation_from_matrix_v1(context(), case_set_id, &matrix(false, true)).unwrap();
        assert_eq!(observation.disposition, RunDisposition::Completed);
        assert_eq!(observation.failures.len(), 1);
        assert_eq!(observation.failures[0].category, "fall");
    }
}
