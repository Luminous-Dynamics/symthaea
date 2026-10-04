// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Passive-function design contracts.
//!
//! This module makes a useful engineering distinction explicit:
//!
//! A design may have dynamic behaviour without requiring mechanically moving
//! solid parts or active control hardware.
//!
//! Examples include geometry-controlled fluidic rectifiers, phase-change
//! thermal devices, resonant/acoustic structures, passive electromagnetic
//! structures, and compliant monolithic materials.
//!
//! The types here do not certify physical behaviour. They provide a
//! deterministic contract that can be attached to a generated design and then
//! checked against CAD, simulation, and manufacturing evidence.

/// Physical domain that can provide an input stimulus to a passive design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassiveInput {
    Mechanical,
    Fluidic,
    Thermal,
    Acoustic,
    Electromagnetic,
    Optical,
    Chemical,
}

/// Physical domain in which a passive design produces a useful output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassiveOutput {
    Mechanical,
    Fluidic,
    Thermal,
    Acoustic,
    Electromagnetic,
    Optical,
    Chemical,
}

/// Broad physical strategy used to obtain function without an active actuator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassiveMechanism {
    Geometry,
    MaterialResponse,
    PhaseChange,
    DistributedTransport,
    WaveInteraction,
    FieldInteraction,
    Hybrid,
}

/// Conservative policy for a strict no-moving-mechanical-parts design.
///
/// The policy permits changing states of matter or fields, because otherwise
/// useful passive devices such as heat pipes, fluidic rectifiers, acoustic
/// resonators, and compliant monolithic structures would be excluded despite
/// having no mechanically articulated parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoMovingPartsPolicy {
    pub max_moving_solid_components: usize,
    pub max_mechanical_joints: usize,
    pub max_active_power_w: f64,
    pub allow_commanded_actuators: bool,
    pub allow_external_control: bool,
    pub allow_fluid_motion: bool,
    pub allow_distributed_deformation: bool,
    pub allow_phase_change: bool,
}

impl Default for NoMovingPartsPolicy {
    fn default() -> Self {
        Self {
            max_moving_solid_components: 0,
            max_mechanical_joints: 0,
            max_active_power_w: 0.0,
            allow_commanded_actuators: false,
            allow_external_control: false,
            allow_fluid_motion: true,
            allow_distributed_deformation: true,
            allow_phase_change: true,
        }
    }
}

/// Evidence extracted from CAD, simulation, or a manufacturing description.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveDesignEvidence {
    pub moving_solid_components: usize,
    pub mechanical_joints: usize,
    pub active_power_w: f64,
    pub commanded_actuators: usize,
    pub requires_external_control: bool,
    pub uses_fluid_motion: bool,
    pub uses_distributed_deformation: bool,
    pub uses_phase_change: bool,
}

impl Default for PassiveDesignEvidence {
    fn default() -> Self {
        Self {
            moving_solid_components: 0,
            mechanical_joints: 0,
            active_power_w: 0.0,
            commanded_actuators: 0,
            requires_external_control: false,
            uses_fluid_motion: false,
            uses_distributed_deformation: false,
            uses_phase_change: false,
        }
    }
}

/// A physical design intent expressed as a passive function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveFunctionContract {
    pub input: PassiveInput,
    pub output: PassiveOutput,
    pub mechanism: PassiveMechanism,
    pub policy: NoMovingPartsPolicy,
}

impl PassiveFunctionContract {
    /// Construct a strict no-moving-parts contract with the default policy.
    pub fn strict(
        input: PassiveInput,
        output: PassiveOutput,
        mechanism: PassiveMechanism,
    ) -> Self {
        Self {
            input,
            output,
            mechanism,
            policy: NoMovingPartsPolicy::default(),
        }
    }

    /// Validate evidence against the passive-design contract.
    pub fn validate(&self, evidence: PassiveDesignEvidence) -> PassiveValidationReport {
        let mut violations = Vec::new();

        if evidence.moving_solid_components > self.policy.max_moving_solid_components {
            violations.push(PassiveViolation::MovingSolidComponents {
                observed: evidence.moving_solid_components,
                allowed: self.policy.max_moving_solid_components,
            });
        }

        if evidence.mechanical_joints > self.policy.max_mechanical_joints {
            violations.push(PassiveViolation::MechanicalJoints {
                observed: evidence.mechanical_joints,
                allowed: self.policy.max_mechanical_joints,
            });
        }

        if !evidence.active_power_w.is_finite() || evidence.active_power_w < 0.0 {
            violations.push(PassiveViolation::InvalidActivePower);
        } else if evidence.active_power_w > self.policy.max_active_power_w + 1.0e-12 {
            violations.push(PassiveViolation::ActivePower {
                observed_w: evidence.active_power_w,
                allowed_w: self.policy.max_active_power_w,
            });
        }

        if !self.policy.allow_commanded_actuators && evidence.commanded_actuators != 0 {
            violations.push(PassiveViolation::CommandedActuators {
                observed: evidence.commanded_actuators,
            });
        }

        if !self.policy.allow_external_control && evidence.requires_external_control {
            violations.push(PassiveViolation::ExternalControlRequired);
        }

        if evidence.uses_fluid_motion && !self.policy.allow_fluid_motion {
            violations.push(PassiveViolation::FluidMotionNotAllowed);
        }

        if evidence.uses_distributed_deformation && !self.policy.allow_distributed_deformation {
            violations.push(PassiveViolation::DistributedDeformationNotAllowed);
        }

        if evidence.uses_phase_change && !self.policy.allow_phase_change {
            violations.push(PassiveViolation::PhaseChangeNotAllowed);
        }

        let compliant = violations.is_empty();

        PassiveValidationReport {
            contract: *self,
            evidence,
            compliant,
            score: self.passivity_score(&evidence),
            violations,
        }
    }

    /// Continuous score in [0, 1] for multi-objective search.
    ///
    /// A score of 1.0 means the candidate satisfies the contract exactly.
    /// The score is not a proof: a high score can still correspond to an
    /// incomplete or incorrect physical model.
    pub fn passivity_score(&self, evidence: &PassiveDesignEvidence) -> f64 {
        if !evidence.active_power_w.is_finite() || evidence.active_power_w < 0.0 {
            return 0.0;
        }

        let mut score: f64 = 1.0;

        if evidence.moving_solid_components > self.policy.max_moving_solid_components {
            let ratio = evidence.moving_solid_components as f64
                / self.policy.max_moving_solid_components.max(1) as f64;
            score *= 1.0 / (1.0 + ratio);
        }

        if evidence.mechanical_joints > self.policy.max_mechanical_joints {
            let ratio =
                evidence.mechanical_joints as f64 / self.policy.max_mechanical_joints.max(1) as f64;
            score *= 1.0 / (1.0 + ratio);
        }

        if evidence.active_power_w > self.policy.max_active_power_w {
            let denominator = self.policy.max_active_power_w.max(1.0);
            score *= 1.0 / (1.0 + evidence.active_power_w / denominator);
        }

        if !self.policy.allow_commanded_actuators && evidence.commanded_actuators > 0 {
            score *= 0.5_f64.powi(evidence.commanded_actuators.min(8) as i32);
        }

        if !self.policy.allow_external_control && evidence.requires_external_control {
            score *= 0.5;
        }

        if evidence.uses_fluid_motion && !self.policy.allow_fluid_motion {
            score *= 0.75;
        }

        if evidence.uses_distributed_deformation && !self.policy.allow_distributed_deformation {
            score *= 0.75;
        }

        if evidence.uses_phase_change && !self.policy.allow_phase_change {
            score *= 0.75;
        }

        score.clamp(0.0, 1.0)
    }
}

/// Why a candidate failed a passive-function contract.
#[derive(Debug, Clone, PartialEq)]
pub enum PassiveViolation {
    MovingSolidComponents { observed: usize, allowed: usize },
    MechanicalJoints { observed: usize, allowed: usize },
    InvalidActivePower,
    ActivePower { observed_w: f64, allowed_w: f64 },
    CommandedActuators { observed: usize },
    ExternalControlRequired,
    FluidMotionNotAllowed,
    DistributedDeformationNotAllowed,
    PhaseChangeNotAllowed,
}

/// Deterministic verification result for a passive-function candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct PassiveValidationReport {
    pub contract: PassiveFunctionContract,
    pub evidence: PassiveDesignEvidence,
    pub compliant: bool,
    pub score: f64,
    pub violations: Vec<PassiveViolation>,
}

/// Weights for combining passive compliance with other engineering objectives.
///
/// The weights are deliberately caller-configurable. There is no universally
/// correct scalarization across engineering domains, and the frontier should
/// remain recoverable from the underlying objective observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveObjectiveWeights {
    /// Weight for passivity.
    pub passivity: f64,
    /// Weight for functional/physical performance.
    pub performance: f64,
    /// Weight for manufacturing feasibility.
    pub manufacturability: f64,
    /// Weight for material efficiency.
    pub material_efficiency: f64,
}

impl Default for PassiveObjectiveWeights {
    fn default() -> Self {
        Self {
            passivity: 0.35,
            performance: 0.35,
            manufacturability: 0.20,
            material_efficiency: 0.10,
        }
    }
}

impl PassiveObjectiveWeights {
    /// Return true when all weights are finite, non-negative, and sum to > 0.
    pub fn is_valid(&self) -> bool {
        let values = [
            self.passivity,
            self.performance,
            self.manufacturability,
            self.material_efficiency,
        ];
        values.iter().all(|value| value.is_finite() && *value >= 0.0)
            && values.iter().sum::<f64>() > 0.0
    }

    /// Normalize the weights to unit sum.
    pub fn normalized(self) -> Option<Self> {
        if !self.is_valid() {
            return None;
        }
        let total = self.passivity
            + self.performance
            + self.manufacturability
            + self.material_efficiency;
        Some(Self {
            passivity: self.passivity / total,
            performance: self.performance / total,
            manufacturability: self.manufacturability / total,
            material_efficiency: self.material_efficiency / total,
        })
    }
}

/// Objective observations supplied by domain-specific evaluators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveObjectiveObservation {
    /// Continuous passive-design score in [0, 1].
    pub passivity: f64,
    /// Functional/physical performance score in [0, 1].
    pub performance: f64,
    /// Manufacturing feasibility score in [0, 1].
    pub manufacturability: f64,
    /// Material-efficiency score in [0, 1].
    pub material_efficiency: f64,
}

impl PassiveObjectiveObservation {
    /// Combine normalized observations with configurable weights.
    ///
    /// Invalid observations or invalid weights return 0.0 so a malformed
    /// candidate cannot accidentally receive a positive optimization score.
    pub fn weighted_score(&self, weights: PassiveObjectiveWeights) -> f64 {
        let normalized = match weights.normalized() {
            Some(weights) => weights,
            None => return 0.0,
        };
        let values = [
            self.passivity,
            self.performance,
            self.manufacturability,
            self.material_efficiency,
        ];
        if values.iter().any(|value| !value.is_finite() || !(0.0..=1.0).contains(value)) {
            return 0.0;
        }

        (self.passivity * normalized.passivity
            + self.performance * normalized.performance
            + self.manufacturability * normalized.manufacturability
            + self.material_efficiency * normalized.material_efficiency)
            .clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_strict_but_allows_state_changes() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Thermal,
            PassiveOutput::Thermal,
            PassiveMechanism::PhaseChange,
        );
        let evidence = PassiveDesignEvidence {
            uses_phase_change: true,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(report.compliant);
        assert!((report.score - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn moving_solid_part_breaks_strict_contract() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Fluidic,
            PassiveOutput::Fluidic,
            PassiveMechanism::Geometry,
        );
        let evidence = PassiveDesignEvidence {
            moving_solid_components: 1,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(!report.compliant);
        assert!(report
            .violations
            .iter()
            .any(|v| matches!(v, PassiveViolation::MovingSolidComponents { .. })));
        assert!(report.score < 1.0);
    }

    #[test]
    fn joints_break_strict_contract() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Mechanical,
            PassiveOutput::Mechanical,
            PassiveMechanism::MaterialResponse,
        );
        let evidence = PassiveDesignEvidence {
            mechanical_joints: 1,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(!report.compliant);
    }

    #[test]
    fn active_power_breaks_zero_power_contract() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Electromagnetic,
            PassiveOutput::Optical,
            PassiveMechanism::FieldInteraction,
        );
        let evidence = PassiveDesignEvidence {
            active_power_w: 5.0,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(!report.compliant);
        assert!(report.score < 1.0);
    }

    #[test]
    fn actuator_and_control_are_rejected() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Optical,
            PassiveOutput::Optical,
            PassiveMechanism::Geometry,
        );
        let evidence = PassiveDesignEvidence {
            commanded_actuators: 1,
            requires_external_control: true,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(!report.compliant);
        assert_eq!(report.violations.len(), 2);
    }

    #[test]
    fn distributed_fluidic_design_can_be_fully_passive() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Fluidic,
            PassiveOutput::Fluidic,
            PassiveMechanism::DistributedTransport,
        );
        let evidence = PassiveDesignEvidence {
            uses_fluid_motion: true,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(report.compliant);
        assert_eq!(report.score, 1.0);
    }

    #[test]
    fn default_objective_weights_normalize() {
        let weights = PassiveObjectiveWeights::default();
        let normalized = weights.normalized().expect("default weights are valid");
        let total = normalized.passivity
            + normalized.performance
            + normalized.manufacturability
            + normalized.material_efficiency;
        assert!((total - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_objective_observation_scores_zero() {
        let observation = PassiveObjectiveObservation {
            passivity: f64::NAN,
            performance: 0.9,
            manufacturability: 0.9,
            material_efficiency: 0.9,
        };
        assert_eq!(
            observation.weighted_score(PassiveObjectiveWeights::default()),
            0.0
        );
    }

    #[test]
    fn objective_score_is_bounded_and_weighted() {
        let observation = PassiveObjectiveObservation {
            passivity: 1.0,
            performance: 0.5,
            manufacturability: 0.5,
            material_efficiency: 0.0,
        };
        let score = observation.weighted_score(PassiveObjectiveWeights::default());
        assert!((0.0..=1.0).contains(&score));
        assert!((score - 0.7).abs() < 1.0e-12);
    }

    #[test]
    fn non_finite_power_is_not_accepted() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Thermal,
            PassiveOutput::Mechanical,
            PassiveMechanism::MaterialResponse,
        );
        let evidence = PassiveDesignEvidence {
            active_power_w: f64::NAN,
            ..Default::default()
        };

        let report = contract.validate(evidence);
        assert!(!report.compliant);
        assert!(report.violations.contains(&PassiveViolation::InvalidActivePower));
        assert_eq!(report.score, 0.0);
    }
}
