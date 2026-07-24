// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic push-recovery benchmark for the stabilizing control substrate.

use serde::{Deserialize, Serialize};

use crate::actuation::ActuationAdapter;
use crate::hierarchical::HierarchicalHumanoidController;
use crate::recovery::RecoveryMode;
use crate::safety::HumanoidSafetyProjector;
use crate::simulator::HumanoidPhysicsSimulator;
use crate::types::{HumanoidCommand, HumanoidPdGains, HumanoidTask, pd_standing_baseline};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRecoveryProtocol {
    pub physics_hz: f64,
    pub settle_seconds: f64,
    pub evaluate_seconds: f64,
    pub push_force_n: [f64; 3],
    pub recovered_margin_m: f64,
    pub recovered_uprightness: f64,
    pub recovered_hold_seconds: f64,
}

impl Default for PushRecoveryProtocol {
    fn default() -> Self {
        Self {
            physics_hz: 400.0,
            settle_seconds: 0.75,
            evaluate_seconds: 4.0,
            push_force_n: [180.0, 0.0, 0.0],
            recovered_margin_m: 0.02,
            recovered_uprightness: 0.90,
            recovered_hold_seconds: 0.35,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRecoveryResult {
    pub recovered: bool,
    pub fell: bool,
    pub recovery_time_s: Option<f64>,
    pub min_uprightness: f64,
    pub min_capture_margin_m: f64,
    pub peak_recovery_effort: f32,
    pub recovery_interventions: usize,
}

/// Exercise the deterministic standing, capture-point, and safety layers with a
/// one-tick external push. Learned residual authority is zero, so regressions in
/// the stabilizing substrate cannot be hidden by a trained network.
pub fn run_standing_push_recovery(
    simulator: &mut dyn HumanoidPhysicsSimulator,
    protocol: &PushRecoveryProtocol,
) -> PushRecoveryResult {
    assert!(protocol.physics_hz.is_finite() && protocol.physics_hz > 0.0);
    let dt = 1.0 / protocol.physics_hz;
    let morphology = simulator.morphology();
    let gains = HumanoidPdGains::for_morphology(morphology);
    let hierarchy = HierarchicalHumanoidController::new(morphology);
    let mut safety = HumanoidSafetyProjector::new(morphology);
    let adapter = ActuationAdapter::default();
    let zero = HumanoidCommand::zero_for(morphology.num_actuators());

    simulator.reset();
    let settle_steps = (protocol.settle_seconds * protocol.physics_hz).round() as usize;
    for _ in 0..settle_steps {
        apply_standing_step(
            simulator,
            &gains,
            &hierarchy,
            &mut safety,
            &adapter,
            &zero,
            dt,
        );
    }
    simulator.apply_external_force(protocol.push_force_n);

    let max_steps = (protocol.evaluate_seconds * protocol.physics_hz).round() as usize;
    let hold_steps = (protocol.recovered_hold_seconds * protocol.physics_hz).round() as usize;
    let mut stable_streak = 0usize;
    let mut min_uprightness = f64::INFINITY;
    let mut min_margin = f64::INFINITY;
    let mut peak_effort = 0.0f32;
    let mut interventions = 0usize;
    let mut recovered_at = None;
    let mut fell = false;

    for step in 0..max_steps {
        let report = apply_standing_step(
            simulator,
            &gains,
            &hierarchy,
            &mut safety,
            &adapter,
            &zero,
            dt,
        );
        let state = simulator.true_state();
        min_uprightness = min_uprightness.min(state.uprightness());
        if report.capture_margin_m.is_finite() {
            min_margin = min_margin.min(report.capture_margin_m);
        }
        peak_effort = peak_effort.max(report.recovery_effort);
        if report.recovery_mode != RecoveryMode::Nominal {
            interventions += 1;
        }
        if state.uprightness() < 0.2 || state.head_height < 0.5 {
            fell = true;
            break;
        }
        let stable = state.uprightness() >= protocol.recovered_uprightness
            && report.capture_margin_m >= protocol.recovered_margin_m
            && report.recovery_mode == RecoveryMode::Nominal;
        stable_streak = if stable { stable_streak + 1 } else { 0 };
        if stable_streak >= hold_steps.max(1) {
            recovered_at = Some((step + 1) as f64 * dt);
            break;
        }
    }

    PushRecoveryResult {
        recovered: recovered_at.is_some() && !fell,
        fell,
        recovery_time_s: recovered_at,
        min_uprightness: if min_uprightness.is_finite() {
            min_uprightness
        } else {
            0.0
        },
        min_capture_margin_m: if min_margin.is_finite() {
            min_margin
        } else {
            0.0
        },
        peak_recovery_effort: peak_effort,
        recovery_interventions: interventions,
    }
}

fn apply_standing_step(
    simulator: &mut dyn HumanoidPhysicsSimulator,
    gains: &HumanoidPdGains,
    hierarchy: &HierarchicalHumanoidController,
    safety: &mut HumanoidSafetyProjector,
    adapter: &ActuationAdapter,
    zero: &HumanoidCommand,
    dt: f64,
) -> crate::hierarchical::HierarchicalControlReport {
    let observation = simulator.observation().clone();
    let contacts = simulator.contact_frame();
    let baseline = pd_standing_baseline(&observation, gains);
    let (command, report) = hierarchy.synthesize_with_contacts(
        HumanoidTask::Stand,
        &observation,
        &contacts,
        &baseline,
        zero,
        1.0,
        0.0,
    );
    let projected = safety.project(
        &command,
        &observation,
        crate::types::ActuationMode::NormalizedTorque,
        dt,
    );
    let adapted = adapter
        .adapt_normalized_torque_intent(
            &projected.command,
            &observation,
            simulator.morphology(),
            simulator.actuation_mode(),
        )
        .expect("benchmark command must match validated backend capabilities");
    simulator.step(&adapted.command, dt);
    report
}

/// Cardinal and diagonal disturbance directions used by the recovery matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PushDirection {
    Forward,
    Backward,
    Left,
    Right,
    ForwardLeft,
    ForwardRight,
    BackwardLeft,
    BackwardRight,
}

impl PushDirection {
    pub const ALL: [Self; 8] = [
        Self::Forward,
        Self::Backward,
        Self::Left,
        Self::Right,
        Self::ForwardLeft,
        Self::ForwardRight,
        Self::BackwardLeft,
        Self::BackwardRight,
    ];

    pub fn unit_vector(self) -> [f64; 3] {
        let diagonal = std::f64::consts::FRAC_1_SQRT_2;
        match self {
            Self::Forward => [1.0, 0.0, 0.0],
            Self::Backward => [-1.0, 0.0, 0.0],
            Self::Left => [0.0, 1.0, 0.0],
            Self::Right => [0.0, -1.0, 0.0],
            Self::ForwardLeft => [diagonal, diagonal, 0.0],
            Self::ForwardRight => [diagonal, -diagonal, 0.0],
            Self::BackwardLeft => [-diagonal, diagonal, 0.0],
            Self::BackwardRight => [-diagonal, -diagonal, 0.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRecoveryCaseResult {
    pub direction: PushDirection,
    pub force_n: f64,
    pub result: PushRecoveryResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushRecoveryMatrixResult {
    pub cases: Vec<PushRecoveryCaseResult>,
    pub recovery_rate: f64,
    pub fall_rate: f64,
    pub worst_uprightness: f64,
    pub worst_capture_margin_m: f64,
    pub mean_recovery_time_s: Option<f64>,
    pub directional_asymmetry: f64,
}

/// Evaluate recovery across directions and force magnitudes. A fresh simulator
/// reset is performed for every case by `run_standing_push_recovery`.
pub fn run_push_recovery_matrix(
    simulator: &mut dyn HumanoidPhysicsSimulator,
    template: &PushRecoveryProtocol,
    force_levels_n: &[f64],
) -> PushRecoveryMatrixResult {
    let mut cases = Vec::with_capacity(PushDirection::ALL.len() * force_levels_n.len());
    for force_n in force_levels_n.iter().copied() {
        assert!(force_n.is_finite() && force_n >= 0.0);
        for direction in PushDirection::ALL {
            let unit = direction.unit_vector();
            let mut protocol = template.clone();
            protocol.push_force_n = [unit[0] * force_n, unit[1] * force_n, 0.0];
            cases.push(PushRecoveryCaseResult {
                direction,
                force_n,
                result: run_standing_push_recovery(simulator, &protocol),
            });
        }
    }

    let count = cases.len().max(1) as f64;
    let recovered = cases.iter().filter(|case| case.result.recovered).count();
    let falls = cases.iter().filter(|case| case.result.fell).count();
    let worst_uprightness = cases
        .iter()
        .map(|case| case.result.min_uprightness)
        .fold(f64::INFINITY, f64::min);
    let worst_capture_margin_m = cases
        .iter()
        .map(|case| case.result.min_capture_margin_m)
        .fold(f64::INFINITY, f64::min);
    let recovery_times: Vec<f64> = cases
        .iter()
        .filter_map(|case| case.result.recovery_time_s)
        .collect();
    let mean_recovery_time_s = if recovery_times.is_empty() {
        None
    } else {
        Some(recovery_times.iter().sum::<f64>() / recovery_times.len() as f64)
    };
    let directional_asymmetry = directional_asymmetry(&cases);

    PushRecoveryMatrixResult {
        cases,
        recovery_rate: recovered as f64 / count,
        fall_rate: falls as f64 / count,
        worst_uprightness: if worst_uprightness.is_finite() {
            worst_uprightness
        } else {
            0.0
        },
        worst_capture_margin_m: if worst_capture_margin_m.is_finite() {
            worst_capture_margin_m
        } else {
            0.0
        },
        mean_recovery_time_s,
        directional_asymmetry,
    }
}

fn directional_asymmetry(cases: &[PushRecoveryCaseResult]) -> f64 {
    fn recovery_score(result: &PushRecoveryResult) -> f64 {
        if result.recovered {
            1.0
        } else if result.fell {
            0.0
        } else {
            0.5
        }
    }
    let pairs = [
        (PushDirection::Forward, PushDirection::Backward),
        (PushDirection::Left, PushDirection::Right),
        (PushDirection::ForwardLeft, PushDirection::BackwardRight),
        (PushDirection::ForwardRight, PushDirection::BackwardLeft),
    ];
    let mut differences = Vec::new();
    for (a, b) in pairs {
        let a_scores: Vec<f64> = cases
            .iter()
            .filter(|case| case.direction == a)
            .map(|case| recovery_score(&case.result))
            .collect();
        let b_scores: Vec<f64> = cases
            .iter()
            .filter(|case| case.direction == b)
            .map(|case| recovery_score(&case.result))
            .collect();
        for (left, right) in a_scores.iter().zip(b_scores.iter()) {
            differences.push((left - right).abs());
        }
    }
    if differences.is_empty() {
        0.0
    } else {
        differences.iter().sum::<f64>() / differences.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::SimpleHumanoidSimulator;

    #[test]
    fn zero_push_protocol_is_recoverable() {
        let mut simulator = SimpleHumanoidSimulator::new().with_ground_contact(true);
        let protocol = PushRecoveryProtocol {
            push_force_n: [0.0; 3],
            evaluate_seconds: 1.0,
            ..PushRecoveryProtocol::default()
        };
        let result = run_standing_push_recovery(&mut simulator, &protocol);
        assert!(!result.fell);
    }

    #[test]
    fn recovery_matrix_covers_all_directions() {
        let mut simulator = SimpleHumanoidSimulator::new().with_ground_contact(true);
        let protocol = PushRecoveryProtocol {
            evaluate_seconds: 0.05,
            settle_seconds: 0.0,
            ..PushRecoveryProtocol::default()
        };
        let matrix = run_push_recovery_matrix(&mut simulator, &protocol, &[0.0]);
        assert_eq!(matrix.cases.len(), PushDirection::ALL.len());
        assert!((0.0..=1.0).contains(&matrix.directional_asymmetry));
    }
}
