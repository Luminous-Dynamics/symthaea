// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Low-gravity workout benchmark for SX-009.
//!
//! The benchmark compares exercise modalities against the same declared
//! physiological/loading target. Matching a target is not evidence that the
//! modalities are biologically equivalent; human-in-the-loop evidence is still
//! required for that claim.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkoutModality {
    OrdinaryReducedGravityMovement,
    ExosuitResistance,
    ConventionalExerciseHardware,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkoutProtocol {
    pub protocol_id: String,
    pub duration_min: f64,
    pub target_metabolic_power_w: f64,
    pub metabolic_tolerance_fraction: f64,
    /// Target integrated external musculoskeletal loading proxy, N*s.
    pub target_load_impulse_ns: f64,
    pub load_tolerance_fraction: f64,
    pub evidence_required: ExosuitEvidenceLevel,
}

impl WorkoutProtocol {
    pub fn is_valid(&self) -> bool {
        !self.protocol_id.trim().is_empty()
            && self.duration_min.is_finite()
            && self.duration_min > 0.0
            && self.target_metabolic_power_w.is_finite()
            && self.target_metabolic_power_w > 0.0
            && self.metabolic_tolerance_fraction.is_finite()
            && (0.0..=1.0).contains(&self.metabolic_tolerance_fraction)
            && self.target_load_impulse_ns.is_finite()
            && self.target_load_impulse_ns > 0.0
            && self.load_tolerance_fraction.is_finite()
            && (0.0..=1.0).contains(&self.load_tolerance_fraction)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkoutMeasurement {
    pub protocol_id: String,
    pub modality: WorkoutModality,
    pub metabolic_power_w: f64,
    pub load_impulse_ns: f64,
    pub electrical_energy_wh: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl WorkoutMeasurement {
    pub fn is_valid(&self) -> bool {
        !self.protocol_id.trim().is_empty()
            && [
                self.metabolic_power_w,
                self.load_impulse_ns,
                self.electrical_energy_wh,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkoutGateOutcome {
    Pass,
    MissesMetabolicTarget,
    MissesLoadingTarget,
    InsufficientEvidence,
    Incomparable,
}

pub fn evaluate_workout_gate(
    protocol: &WorkoutProtocol,
    measurement: &WorkoutMeasurement,
) -> WorkoutGateOutcome {
    if !protocol.is_valid()
        || !measurement.is_valid()
        || protocol.protocol_id != measurement.protocol_id
    {
        return WorkoutGateOutcome::Incomparable;
    }
    if measurement.evidence < protocol.evidence_required {
        return WorkoutGateOutcome::InsufficientEvidence;
    }

    let metabolic_error =
        (measurement.metabolic_power_w - protocol.target_metabolic_power_w).abs()
            / protocol.target_metabolic_power_w;
    if metabolic_error > protocol.metabolic_tolerance_fraction {
        return WorkoutGateOutcome::MissesMetabolicTarget;
    }

    let loading_error = (measurement.load_impulse_ns - protocol.target_load_impulse_ns).abs()
        / protocol.target_load_impulse_ns;
    if loading_error > protocol.load_tolerance_fraction {
        return WorkoutGateOutcome::MissesLoadingTarget;
    }

    WorkoutGateOutcome::Pass
}

/// Compare energy cost only after two modalities have independently met the
/// same physiological/loading gate.
pub fn lower_energy_equivalent<'a>(
    protocol: &WorkoutProtocol,
    a: &'a WorkoutMeasurement,
    b: &'a WorkoutMeasurement,
) -> Option<&'a WorkoutMeasurement> {
    if evaluate_workout_gate(protocol, a) != WorkoutGateOutcome::Pass
        || evaluate_workout_gate(protocol, b) != WorkoutGateOutcome::Pass
    {
        return None;
    }
    if a.electrical_energy_wh <= b.electrical_energy_wh {
        Some(a)
    } else {
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol() -> WorkoutProtocol {
        WorkoutProtocol {
            protocol_id: "lunar-strength-v1".into(),
            duration_min: 30.0,
            target_metabolic_power_w: 400.0,
            metabolic_tolerance_fraction: 0.10,
            target_load_impulse_ns: 100_000.0,
            load_tolerance_fraction: 0.10,
            evidence_required: ExosuitEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn ordinary_lunar_movement_can_miss_loading_target() {
        let m = WorkoutMeasurement {
            protocol_id: "lunar-strength-v1".into(),
            modality: WorkoutModality::OrdinaryReducedGravityMovement,
            metabolic_power_w: 390.0,
            load_impulse_ns: 40_000.0,
            electrical_energy_wh: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        assert_eq!(
            evaluate_workout_gate(&protocol(), &m),
            WorkoutGateOutcome::MissesLoadingTarget
        );
    }

    #[test]
    fn resistance_mode_can_match_declared_target() {
        let m = WorkoutMeasurement {
            protocol_id: "lunar-strength-v1".into(),
            modality: WorkoutModality::ExosuitResistance,
            metabolic_power_w: 410.0,
            load_impulse_ns: 102_000.0,
            electrical_energy_wh: 35.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        assert_eq!(evaluate_workout_gate(&protocol(), &m), WorkoutGateOutcome::Pass);
    }

    #[test]
    fn qualification_gate_rejects_simulation_result() {
        let mut p = protocol();
        p.evidence_required = ExosuitEvidenceLevel::Qualification;
        let m = WorkoutMeasurement {
            protocol_id: "lunar-strength-v1".into(),
            modality: WorkoutModality::ExosuitResistance,
            metabolic_power_w: 400.0,
            load_impulse_ns: 100_000.0,
            electrical_energy_wh: 30.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        assert_eq!(
            evaluate_workout_gate(&p, &m),
            WorkoutGateOutcome::InsufficientEvidence
        );
    }
}
