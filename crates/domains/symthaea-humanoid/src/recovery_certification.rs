// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic adversarial fall and recovery certification.
//!
//! Certification expands the directional push matrix across disturbance timing
//! and force. It records every case, checks directional symmetry and force
//! monotonicity, and emits an explicit pass/fail artifact. This is simulation
//! evidence only; it does not certify physical hardware safety.

use serde::{Deserialize, Serialize};

use crate::recovery_benchmark::{
    PushDirection, PushRecoveryCaseResult, PushRecoveryProtocol, run_push_recovery_matrix,
};
use crate::simulator::HumanoidPhysicsSimulator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialRecoveryCertificationConfig {
    pub force_levels_n: Vec<f64>,
    pub disturbance_times_s: Vec<f64>,
    pub minimum_recovery_rate: f64,
    pub maximum_fall_rate: f64,
    pub maximum_directional_asymmetry: f64,
    pub minimum_uprightness: f64,
    pub minimum_capture_margin_m: f64,
    pub maximum_mean_recovery_time_s: f64,
    pub maximum_monotonicity_violations: usize,
}

impl Default for AdversarialRecoveryCertificationConfig {
    fn default() -> Self {
        Self {
            force_levels_n: vec![0.0, 80.0, 140.0, 200.0],
            disturbance_times_s: vec![0.0, 0.20, 0.50],
            minimum_recovery_rate: 0.72,
            maximum_fall_rate: 0.18,
            maximum_directional_asymmetry: 0.30,
            minimum_uprightness: 0.18,
            minimum_capture_margin_m: -0.35,
            maximum_mean_recovery_time_s: 2.5,
            maximum_monotonicity_violations: 0,
        }
    }
}

impl AdversarialRecoveryCertificationConfig {
    pub fn validate(&self) -> bool {
        !self.force_levels_n.is_empty()
            && !self.disturbance_times_s.is_empty()
            && self
                .force_levels_n
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self
                .disturbance_times_s
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && (0.0..=1.0).contains(&self.minimum_recovery_rate)
            && (0.0..=1.0).contains(&self.maximum_fall_rate)
            && (0.0..=1.0).contains(&self.maximum_directional_asymmetry)
            && self.minimum_uprightness.is_finite()
            && self.minimum_capture_margin_m.is_finite()
            && self.maximum_mean_recovery_time_s.is_finite()
            && self.maximum_mean_recovery_time_s >= 0.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedRecoveryCase {
    pub disturbance_time_s: f64,
    pub direction: PushDirection,
    pub force_n: f64,
    pub result: crate::recovery_benchmark::PushRecoveryResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialRecoveryCertificate {
    pub schema_version: u32,
    pub protocol_id: String,
    pub scenario_fingerprint: u64,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
    pub cases: Vec<TimedRecoveryCase>,
    pub recovery_rate: f64,
    pub fall_rate: f64,
    pub worst_uprightness: f64,
    pub worst_capture_margin_m: f64,
    pub mean_recovery_time_s: Option<f64>,
    pub maximum_directional_asymmetry: f64,
    pub monotonicity_violations: usize,
}

pub fn certify_adversarial_recovery(
    simulator: &mut dyn HumanoidPhysicsSimulator,
    protocol_template: &PushRecoveryProtocol,
    config: &AdversarialRecoveryCertificationConfig,
) -> AdversarialRecoveryCertificate {
    if !config.validate() {
        return invalid_certificate("invalid adversarial recovery certification configuration");
    }
    if !protocol_is_valid(protocol_template) {
        return invalid_certificate("invalid push-recovery protocol configuration");
    }
    let mut cases = Vec::new();
    let mut maximum_directional_asymmetry = 0.0f64;
    for disturbance_time_s in config.disturbance_times_s.iter().copied() {
        let mut protocol = protocol_template.clone();
        protocol.settle_seconds = disturbance_time_s;
        let matrix = run_push_recovery_matrix(simulator, &protocol, &config.force_levels_n);
        maximum_directional_asymmetry =
            maximum_directional_asymmetry.max(matrix.directional_asymmetry);
        cases.extend(matrix.cases.into_iter().map(
            |PushRecoveryCaseResult {
                 direction,
                 force_n,
                 result,
             }| TimedRecoveryCase {
                disturbance_time_s,
                direction,
                force_n,
                result,
            },
        ));
    }

    let count = cases.len().max(1) as f64;
    let recovery_rate = cases.iter().filter(|case| case.result.recovered).count() as f64 / count;
    let fall_rate = cases.iter().filter(|case| case.result.fell).count() as f64 / count;
    let worst_uprightness = cases
        .iter()
        .map(|case| case.result.min_uprightness)
        .fold(f64::INFINITY, f64::min);
    let worst_capture_margin_m = cases
        .iter()
        .map(|case| case.result.min_capture_margin_m)
        .fold(f64::INFINITY, f64::min);
    let recovery_times = cases
        .iter()
        .filter_map(|case| case.result.recovery_time_s)
        .collect::<Vec<_>>();
    let mean_recovery_time_s = (!recovery_times.is_empty())
        .then(|| recovery_times.iter().sum::<f64>() / recovery_times.len() as f64);
    let monotonicity_violations = count_monotonicity_violations(&cases);

    let mut failure_reasons = Vec::new();
    if recovery_rate < config.minimum_recovery_rate {
        failure_reasons.push(format!(
            "recovery rate {:.3} is below {:.3}",
            recovery_rate, config.minimum_recovery_rate
        ));
    }
    if fall_rate > config.maximum_fall_rate {
        failure_reasons.push(format!(
            "fall rate {:.3} exceeds {:.3}",
            fall_rate, config.maximum_fall_rate
        ));
    }
    if maximum_directional_asymmetry > config.maximum_directional_asymmetry {
        failure_reasons.push(format!(
            "directional asymmetry {:.3} exceeds {:.3}",
            maximum_directional_asymmetry, config.maximum_directional_asymmetry
        ));
    }
    if worst_uprightness < config.minimum_uprightness {
        failure_reasons.push(format!(
            "worst uprightness {:.3} is below {:.3}",
            worst_uprightness, config.minimum_uprightness
        ));
    }
    if worst_capture_margin_m < config.minimum_capture_margin_m {
        failure_reasons.push(format!(
            "worst capture margin {:.3} m is below {:.3} m",
            worst_capture_margin_m, config.minimum_capture_margin_m
        ));
    }
    if mean_recovery_time_s
        .map(|value| value > config.maximum_mean_recovery_time_s)
        .unwrap_or(false)
    {
        failure_reasons.push(format!(
            "mean recovery time {:.3} s exceeds {:.3} s",
            mean_recovery_time_s.unwrap_or(f64::INFINITY),
            config.maximum_mean_recovery_time_s
        ));
    }
    if monotonicity_violations > config.maximum_monotonicity_violations {
        failure_reasons.push(format!(
            "{} force monotonicity violations exceed {}",
            monotonicity_violations, config.maximum_monotonicity_violations
        ));
    }
    if cases.iter().any(|case| {
        !case.result.min_uprightness.is_finite()
            || !case.result.min_capture_margin_m.is_finite()
            || !case.result.peak_recovery_effort.is_finite()
    }) {
        failure_reasons.push("one or more recovery cases produced non-finite evidence".to_string());
    }

    AdversarialRecoveryCertificate {
        schema_version: 1,
        protocol_id: "symthaea.humanoid.adversarial-recovery.v1".to_string(),
        scenario_fingerprint: scenario_fingerprint(config, protocol_template),
        passed: failure_reasons.is_empty(),
        failure_reasons,
        cases,
        recovery_rate,
        fall_rate,
        worst_uprightness: finite_or_zero(worst_uprightness),
        worst_capture_margin_m: finite_or_zero(worst_capture_margin_m),
        mean_recovery_time_s,
        maximum_directional_asymmetry,
        monotonicity_violations,
    }
}

fn count_monotonicity_violations(cases: &[TimedRecoveryCase]) -> usize {
    let mut violations = 0usize;
    for disturbance_time in unique_times(cases) {
        for direction in PushDirection::ALL {
            let mut directional = cases
                .iter()
                .filter(|case| {
                    case.disturbance_time_s == disturbance_time && case.direction == direction
                })
                .collect::<Vec<_>>();
            directional.sort_by(|left, right| {
                left.force_n
                    .partial_cmp(&right.force_n)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut failure_seen = false;
            for case in directional {
                if failure_seen && case.result.recovered {
                    violations += 1;
                }
                failure_seen |= !case.result.recovered;
            }
        }
    }
    violations
}

fn unique_times(cases: &[TimedRecoveryCase]) -> Vec<f64> {
    let mut times = cases
        .iter()
        .map(|case| case.disturbance_time_s)
        .collect::<Vec<_>>();
    times.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-12);
    times
}

fn protocol_is_valid(protocol: &PushRecoveryProtocol) -> bool {
    protocol.physics_hz.is_finite()
        && protocol.physics_hz > 0.0
        && protocol.settle_seconds.is_finite()
        && protocol.settle_seconds >= 0.0
        && protocol.evaluate_seconds.is_finite()
        && protocol.evaluate_seconds > 0.0
        && protocol.recovered_hold_seconds.is_finite()
        && protocol.recovered_hold_seconds >= 0.0
        && protocol.recovered_margin_m.is_finite()
        && protocol.recovered_uprightness.is_finite()
        && (0.0..=1.0).contains(&protocol.recovered_uprightness)
        && protocol.push_force_n.iter().all(|value| value.is_finite())
}

fn scenario_fingerprint(
    config: &AdversarialRecoveryCertificationConfig,
    protocol: &PushRecoveryProtocol,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let scalar_config = [
        config.minimum_recovery_rate,
        config.maximum_fall_rate,
        config.maximum_directional_asymmetry,
        config.minimum_uprightness,
        config.minimum_capture_margin_m,
        config.maximum_mean_recovery_time_s,
        config.maximum_monotonicity_violations as f64,
    ];
    let scalar_protocol = [
        protocol.physics_hz,
        protocol.settle_seconds,
        protocol.evaluate_seconds,
        protocol.recovered_hold_seconds,
        protocol.recovered_uprightness,
        protocol.recovered_margin_m,
        protocol.push_force_n[0],
        protocol.push_force_n[1],
        protocol.push_force_n[2],
    ];
    for value in config
        .force_levels_n
        .iter()
        .chain(config.disturbance_times_s.iter())
        .chain(scalar_config.iter())
        .chain(scalar_protocol.iter())
    {
        hash_f64(&mut hash, *value);
    }
    hash
}

fn hash_f64(hash: &mut u64, value: f64) {
    for byte in value.to_bits().to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn invalid_certificate(reason: &str) -> AdversarialRecoveryCertificate {
    AdversarialRecoveryCertificate {
        schema_version: 1,
        protocol_id: "symthaea.humanoid.adversarial-recovery.v1".to_string(),
        scenario_fingerprint: 0,
        passed: false,
        failure_reasons: vec![reason.to_string()],
        cases: Vec::new(),
        recovery_rate: 0.0,
        fall_rate: 1.0,
        worst_uprightness: 0.0,
        worst_capture_margin_m: 0.0,
        mean_recovery_time_s: None,
        maximum_directional_asymmetry: 1.0,
        monotonicity_violations: 0,
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::SimpleHumanoidSimulator;

    #[test]
    fn certificate_is_versioned_and_fingerprinted() {
        let mut simulator = SimpleHumanoidSimulator::new().with_ground_contact(true);
        let config = AdversarialRecoveryCertificationConfig {
            force_levels_n: vec![0.0],
            disturbance_times_s: vec![0.0],
            minimum_recovery_rate: 0.0,
            maximum_fall_rate: 1.0,
            maximum_directional_asymmetry: 1.0,
            minimum_uprightness: 0.0,
            minimum_capture_margin_m: -10.0,
            maximum_mean_recovery_time_s: 10.0,
            maximum_monotonicity_violations: 8,
        };
        let protocol = PushRecoveryProtocol {
            evaluate_seconds: 0.05,
            recovered_hold_seconds: 0.01,
            ..PushRecoveryProtocol::default()
        };
        let certificate = certify_adversarial_recovery(&mut simulator, &protocol, &config);
        assert_eq!(certificate.schema_version, 1);
        assert_ne!(certificate.scenario_fingerprint, 0);
        assert_eq!(certificate.cases.len(), PushDirection::ALL.len());
    }

    #[test]
    fn invalid_protocol_fails_closed_without_panicking() {
        let mut simulator = SimpleHumanoidSimulator::new();
        let mut protocol = PushRecoveryProtocol::default();
        protocol.physics_hz = 0.0;
        let certificate = certify_adversarial_recovery(
            &mut simulator,
            &protocol,
            &AdversarialRecoveryCertificationConfig::default(),
        );
        assert!(!certificate.passed);
        assert!(certificate.cases.is_empty());
    }

    #[test]
    fn fingerprint_covers_acceptance_thresholds() {
        let protocol = PushRecoveryProtocol::default();
        let config = AdversarialRecoveryCertificationConfig::default();
        let before = scenario_fingerprint(&config, &protocol);
        let mut changed = config.clone();
        changed.maximum_fall_rate += 0.01;
        assert_ne!(before, scenario_fingerprint(&changed, &protocol));
    }

    #[test]
    fn invalid_configuration_fails_closed() {
        let mut simulator = SimpleHumanoidSimulator::new();
        let mut config = AdversarialRecoveryCertificationConfig::default();
        config.force_levels_n.clear();
        let certificate =
            certify_adversarial_recovery(&mut simulator, &PushRecoveryProtocol::default(), &config);
        assert!(!certificate.passed);
        assert!(certificate.cases.is_empty());
    }
}
