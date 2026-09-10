// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Space-exosuit supervisory safety and benchmark foundation.
//!
//! This module deliberately does **not** model a flight-qualified pressure
//! garment or primary life-support system. It defines the boundary around
//! powered assistance: survival-critical pressure/life-support remains
//! independent, while exoskeleton commands are admitted only inside an
//! explicit deterministic envelope.
//!
//! The safety kernel contains no Phi/consciousness input and no learned-model
//! authority. Symthaea may propose assistance elsewhere; this module only
//! clamps or rejects that proposal using explicit state and limits.

use serde::{Deserialize, Serialize};

use crate::types::{ExoskeletonCommand, NUM_ACTUATORS};

/// Evidence strength for suit descriptors and benchmark results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExosuitEvidenceLevel {
    /// Deterministic or stochastic simulation only.
    Simulation,
    /// Physical component/bench evidence without a suited human.
    HardwareBench,
    /// Human-in-the-loop test under a declared protocol.
    HumanInLoop,
    /// Qualification-grade evidence accepted by the applicable program.
    Qualification,
}

/// Explicit airless-surface environment input for exosuit simulations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpaceEnvironmentState {
    /// Local gravitational acceleration, m/s^2.
    pub gravity_m_s2: f64,
    /// Ambient pressure, Pa.
    pub ambient_pressure_pa: f64,
    /// External environment/surface temperature used by the test model, K.
    pub external_temperature_k: f64,
    /// Normalized abrasive-dust exposure assumption in [0, 1].
    pub dust_exposure: f64,
    /// Radiation dose-rate assumption, mGy/h. This is a model input, not a
    /// certification limit.
    pub radiation_mgy_h: f64,
}

impl SpaceEnvironmentState {
    /// Construct a lunar-surface simulation state.
    ///
    /// The caller supplies temperature and radiation assumptions because both
    /// are mission/location/time dependent. The gravity and near-vacuum values
    /// are only reference simulation inputs, not qualification evidence.
    pub fn lunar_surface(external_temperature_k: f64, radiation_mgy_h: f64) -> Self {
        Self {
            gravity_m_s2: 1.62,
            ambient_pressure_pa: 0.0,
            external_temperature_k,
            dust_exposure: 0.0,
            radiation_mgy_h,
        }
    }

    /// Return true when all environment inputs are finite and physically sane
    /// enough for this simplified supervisory model.
    pub fn is_valid(&self) -> bool {
        self.gravity_m_s2.is_finite()
            && self.gravity_m_s2 >= 0.0
            && self.ambient_pressure_pa.is_finite()
            && self.ambient_pressure_pa >= 0.0
            && self.external_temperature_k.is_finite()
            && self.external_temperature_k > 0.0
            && self.dust_exposure.is_finite()
            && (0.0..=1.0).contains(&self.dust_exposure)
            && self.radiation_mgy_h.is_finite()
            && self.radiation_mgy_h >= 0.0
    }
}

/// Survival/physiology signals exposed *to* the powered-assist supervisor.
///
/// These fields are observations only. This module never commands oxygen,
/// pressure, thermal-control, or other primary life-support hardware.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitSafetyState {
    /// Suit pressure, Pa.
    pub suit_pressure_pa: f64,
    /// Oxygen partial pressure, Pa.
    pub oxygen_partial_pressure_pa: f64,
    /// Carbon-dioxide partial pressure, Pa.
    pub co2_partial_pressure_pa: f64,
    /// Wearer core-temperature estimate, K.
    pub wearer_core_temperature_k: f64,
    /// Powered-assist battery state of charge in [0, 1].
    pub assist_battery_soc: f64,
    /// Pressure-boundary integrity estimate in [0, 1].
    pub pressure_integrity: f64,
    /// Powered-assist electronics health in [0, 1].
    pub assist_electronics_health: f64,
}

impl SuitSafetyState {
    /// Structural validation only; operational acceptability is decided by
    /// `CertifiedAssistEnvelope`.
    pub fn is_finite_and_normalized(&self) -> bool {
        [
            self.suit_pressure_pa,
            self.oxygen_partial_pressure_pa,
            self.co2_partial_pressure_pa,
            self.wearer_core_temperature_k,
            self.assist_battery_soc,
            self.pressure_integrity,
            self.assist_electronics_health,
        ]
        .into_iter()
        .all(f64::is_finite)
            && (0.0..=1.0).contains(&self.assist_battery_soc)
            && (0.0..=1.0).contains(&self.pressure_integrity)
            && (0.0..=1.0).contains(&self.assist_electronics_health)
            && self.suit_pressure_pa >= 0.0
            && self.oxygen_partial_pressure_pa >= 0.0
            && self.co2_partial_pressure_pa >= 0.0
            && self.wearer_core_temperature_k > 0.0
    }
}

/// Deterministic limits within which powered assistance may operate.
///
/// Values must come from a declared test/qualification program. The
/// `simulation_reference()` constructor exists only to exercise software and
/// must never be interpreted as a human-rating specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CertifiedAssistEnvelope {
    /// Per-joint absolute normalized command ceiling in [0, 1].
    pub max_abs_joint_command: [f32; NUM_ACTUATORS],
    pub max_stiffness_gain: f32,
    pub max_damping_gain: f32,
    pub min_suit_pressure_pa: f64,
    pub max_suit_pressure_pa: f64,
    pub min_oxygen_partial_pressure_pa: f64,
    pub max_oxygen_partial_pressure_pa: f64,
    pub max_co2_partial_pressure_pa: f64,
    pub min_core_temperature_k: f64,
    pub max_core_temperature_k: f64,
    pub min_assist_battery_soc: f64,
    pub min_pressure_integrity: f64,
    pub min_assist_electronics_health: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl CertifiedAssistEnvelope {
    /// Illustrative simulation-only envelope for software tests.
    ///
    /// The numerical values intentionally carry `Simulation` evidence and are
    /// not claims about AxEMU, EMU, or any human-rated suit.
    pub fn simulation_reference() -> Self {
        Self {
            max_abs_joint_command: [0.35; NUM_ACTUATORS],
            max_stiffness_gain: 0.5,
            max_damping_gain: 0.5,
            min_suit_pressure_pa: 20_000.0,
            max_suit_pressure_pa: 60_000.0,
            min_oxygen_partial_pressure_pa: 15_000.0,
            max_oxygen_partial_pressure_pa: 40_000.0,
            max_co2_partial_pressure_pa: 1_000.0,
            min_core_temperature_k: 307.0,
            max_core_temperature_k: 312.0,
            min_assist_battery_soc: 0.10,
            min_pressure_integrity: 0.95,
            min_assist_electronics_health: 0.80,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.max_abs_joint_command
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && self.max_stiffness_gain.is_finite()
            && (0.0..=1.0).contains(&self.max_stiffness_gain)
            && self.max_damping_gain.is_finite()
            && (0.0..=1.0).contains(&self.max_damping_gain)
            && self.min_suit_pressure_pa.is_finite()
            && self.max_suit_pressure_pa.is_finite()
            && self.min_suit_pressure_pa <= self.max_suit_pressure_pa
            && self.min_oxygen_partial_pressure_pa.is_finite()
            && self.max_oxygen_partial_pressure_pa.is_finite()
            && self.min_oxygen_partial_pressure_pa <= self.max_oxygen_partial_pressure_pa
            && self.max_co2_partial_pressure_pa.is_finite()
            && self.max_co2_partial_pressure_pa >= 0.0
            && self.min_core_temperature_k.is_finite()
            && self.max_core_temperature_k.is_finite()
            && self.min_core_temperature_k <= self.max_core_temperature_k
            && self.min_assist_battery_soc.is_finite()
            && (0.0..=1.0).contains(&self.min_assist_battery_soc)
            && self.min_pressure_integrity.is_finite()
            && (0.0..=1.0).contains(&self.min_pressure_integrity)
            && self.min_assist_electronics_health.is_finite()
            && (0.0..=1.0).contains(&self.min_assist_electronics_health)
    }
}

/// Why powered assistance was denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistDenialReason {
    InvalidEnvelope,
    InvalidEnvironment,
    InvalidSuitState,
    InvalidCommand,
    SuitPressureOutsideEnvelope,
    OxygenOutsideEnvelope,
    Co2AboveCeiling,
    CoreTemperatureOutsideEnvelope,
    AssistBatteryBelowFloor,
    PressureIntegrityBelowFloor,
    AssistElectronicsBelowFloor,
}

/// Result of the deterministic powered-assist preflight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistDecision {
    /// True only when the proposed command may proceed to the existing local
    /// exoskeleton safety/controller path.
    pub permitted: bool,
    /// Command clamped to the declared assist envelope, or transparent/no
    /// powered assistance when denied.
    pub command: ExoskeletonCommand,
    pub reasons: Vec<AssistDenialReason>,
}

impl AssistDecision {
    fn denied(reasons: Vec<AssistDenialReason>) -> Self {
        Self {
            permitted: false,
            command: transparent_command(),
            reasons,
        }
    }
}

/// Deterministic supervisor for powered-assist admission.
///
/// There is intentionally no consciousness/Phi argument here. This is an
/// independent safety boundary, not a cognition-quality gate.
#[derive(Debug, Clone, Copy)]
pub struct SpaceExosuitSafetyKernel {
    envelope: CertifiedAssistEnvelope,
}

impl SpaceExosuitSafetyKernel {
    pub fn new(envelope: CertifiedAssistEnvelope) -> Self {
        Self { envelope }
    }

    pub fn envelope(&self) -> &CertifiedAssistEnvelope {
        &self.envelope
    }

    /// Validate state, then clamp an assistance request to the certified
    /// envelope. State violations fail closed to a backdrivable/transparent
    /// powered-assist command. Survival systems are not commanded here.
    pub fn evaluate(
        &self,
        environment: &SpaceEnvironmentState,
        suit: &SuitSafetyState,
        proposed: &ExoskeletonCommand,
    ) -> AssistDecision {
        let mut reasons = Vec::new();

        if !self.envelope.is_valid() {
            reasons.push(AssistDenialReason::InvalidEnvelope);
        }
        if !environment.is_valid() {
            reasons.push(AssistDenialReason::InvalidEnvironment);
        }
        if !suit.is_finite_and_normalized() {
            reasons.push(AssistDenialReason::InvalidSuitState);
        }
        if !command_is_finite(proposed) {
            reasons.push(AssistDenialReason::InvalidCommand);
        }

        // Avoid making interval comparisons on malformed state/envelope values.
        if !reasons.is_empty() {
            return AssistDecision::denied(reasons);
        }

        if !(self.envelope.min_suit_pressure_pa..=self.envelope.max_suit_pressure_pa)
            .contains(&suit.suit_pressure_pa)
        {
            reasons.push(AssistDenialReason::SuitPressureOutsideEnvelope);
        }
        if !(self.envelope.min_oxygen_partial_pressure_pa
            ..=self.envelope.max_oxygen_partial_pressure_pa)
            .contains(&suit.oxygen_partial_pressure_pa)
        {
            reasons.push(AssistDenialReason::OxygenOutsideEnvelope);
        }
        if suit.co2_partial_pressure_pa > self.envelope.max_co2_partial_pressure_pa {
            reasons.push(AssistDenialReason::Co2AboveCeiling);
        }
        if !(self.envelope.min_core_temperature_k..=self.envelope.max_core_temperature_k)
            .contains(&suit.wearer_core_temperature_k)
        {
            reasons.push(AssistDenialReason::CoreTemperatureOutsideEnvelope);
        }
        if suit.assist_battery_soc < self.envelope.min_assist_battery_soc {
            reasons.push(AssistDenialReason::AssistBatteryBelowFloor);
        }
        if suit.pressure_integrity < self.envelope.min_pressure_integrity {
            reasons.push(AssistDenialReason::PressureIntegrityBelowFloor);
        }
        if suit.assist_electronics_health < self.envelope.min_assist_electronics_health {
            reasons.push(AssistDenialReason::AssistElectronicsBelowFloor);
        }

        if !reasons.is_empty() {
            return AssistDecision::denied(reasons);
        }

        let mut command = *proposed;
        for (joint, limit) in command
            .joint_torques
            .iter_mut()
            .zip(self.envelope.max_abs_joint_command)
        {
            *joint = joint.clamp(-limit, limit);
        }
        command.stiffness_gain = command
            .stiffness_gain
            .clamp(0.0, self.envelope.max_stiffness_gain);
        command.damping_gain = command
            .damping_gain
            .clamp(0.0, self.envelope.max_damping_gain);

        AssistDecision {
            permitted: true,
            command,
            reasons,
        }
    }
}

fn transparent_command() -> ExoskeletonCommand {
    ExoskeletonCommand {
        joint_torques: [0.0; NUM_ACTUATORS],
        stiffness_gain: 0.0,
        damping_gain: 0.0,
    }
}

fn command_is_finite(command: &ExoskeletonCommand) -> bool {
    command.joint_torques.iter().all(|v| v.is_finite())
        && command.stiffness_gain.is_finite()
        && command.damping_gain.is_finite()
}

/// Benchmark dimensions required before an integrated superiority claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpaceExosuitBenchmarkGate {
    MobilityWorkload,
    Dexterity,
    FailSafeMobility,
    BalanceRecovery,
    Personalization,
    FaultDiagnostics,
    Serviceability,
    DustRobustness,
    ThermalManagement,
    IntegratedEva,
}

/// Whether lower or higher values represent better performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

/// One apples-to-apples candidate-vs-baseline measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    pub gate: SpaceExosuitBenchmarkGate,
    /// Same protocol identifier must apply to both candidate and baseline.
    pub protocol_id: String,
    pub baseline_system: String,
    pub candidate_system: String,
    pub baseline_value: f64,
    pub candidate_value: f64,
    /// 1-sigma or otherwise protocol-declared absolute uncertainty. The exact
    /// statistical interpretation belongs to `protocol_id`.
    pub baseline_uncertainty: f64,
    pub candidate_uncertainty: f64,
    pub direction: MetricDirection,
    pub evidence: ExosuitEvidenceLevel,
}

/// Evaluation of one benchmark gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BenchmarkOutcome {
    Pass { conservative_improvement: f64 },
    Fail { conservative_improvement: f64 },
    InsufficientEvidence,
    Incomparable,
}

impl BenchmarkMeasurement {
    /// Evaluate a candidate relative to a baseline using a conservative
    /// uncertainty-adjusted improvement fraction.
    ///
    /// A 0.10 required improvement means the candidate's conservative bound
    /// must beat the baseline's conservative bound by at least 10% of the
    /// baseline magnitude.
    pub fn evaluate(
        &self,
        minimum_evidence: ExosuitEvidenceLevel,
        required_improvement_fraction: f64,
    ) -> BenchmarkOutcome {
        if self.protocol_id.trim().is_empty()
            || self.baseline_system.trim().is_empty()
            || self.candidate_system.trim().is_empty()
            || self.baseline_system == self.candidate_system
            || !self.baseline_value.is_finite()
            || !self.candidate_value.is_finite()
            || !self.baseline_uncertainty.is_finite()
            || !self.candidate_uncertainty.is_finite()
            || self.baseline_uncertainty < 0.0
            || self.candidate_uncertainty < 0.0
            || !required_improvement_fraction.is_finite()
            || required_improvement_fraction < 0.0
            || self.baseline_value.abs() <= f64::EPSILON
        {
            return BenchmarkOutcome::Incomparable;
        }
        if self.evidence < minimum_evidence {
            return BenchmarkOutcome::InsufficientEvidence;
        }

        let baseline_scale = self.baseline_value.abs();
        let conservative_improvement = match self.direction {
            MetricDirection::LowerIsBetter => {
                let baseline_favorable_bound = self.baseline_value - self.baseline_uncertainty;
                let candidate_adverse_bound = self.candidate_value + self.candidate_uncertainty;
                (baseline_favorable_bound - candidate_adverse_bound) / baseline_scale
            }
            MetricDirection::HigherIsBetter => {
                let baseline_favorable_bound = self.baseline_value + self.baseline_uncertainty;
                let candidate_adverse_bound = self.candidate_value - self.candidate_uncertainty;
                (candidate_adverse_bound - baseline_favorable_bound) / baseline_scale
            }
        };

        if conservative_improvement >= required_improvement_fraction {
            BenchmarkOutcome::Pass {
                conservative_improvement,
            }
        } else {
            BenchmarkOutcome::Fail {
                conservative_improvement,
            }
        }
    }
}

/// Result of checking whether an integrated "better than baseline" claim is
/// supported across all mandatory gates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimReadiness {
    Ready,
    MissingGate(SpaceExosuitBenchmarkGate),
    GateNotPassed(SpaceExosuitBenchmarkGate),
}

/// Check a benchmark campaign for claim readiness.
///
/// This deliberately requires qualification evidence on every mandatory gate;
/// simulation and bench work can guide development but cannot establish an
/// integrated human-rated superiority claim.
pub fn superiority_claim_readiness(
    measurements: &[BenchmarkMeasurement],
    required_improvement_fraction: f64,
) -> ClaimReadiness {
    const REQUIRED: [SpaceExosuitBenchmarkGate; 10] = [
        SpaceExosuitBenchmarkGate::MobilityWorkload,
        SpaceExosuitBenchmarkGate::Dexterity,
        SpaceExosuitBenchmarkGate::FailSafeMobility,
        SpaceExosuitBenchmarkGate::BalanceRecovery,
        SpaceExosuitBenchmarkGate::Personalization,
        SpaceExosuitBenchmarkGate::FaultDiagnostics,
        SpaceExosuitBenchmarkGate::Serviceability,
        SpaceExosuitBenchmarkGate::DustRobustness,
        SpaceExosuitBenchmarkGate::ThermalManagement,
        SpaceExosuitBenchmarkGate::IntegratedEva,
    ];

    for gate in REQUIRED {
        let Some(measurement) = measurements.iter().find(|m| m.gate == gate) else {
            return ClaimReadiness::MissingGate(gate);
        };
        if !matches!(
            measurement.evaluate(
                ExosuitEvidenceLevel::Qualification,
                required_improvement_fraction,
            ),
            BenchmarkOutcome::Pass { .. }
        ) {
            return ClaimReadiness::GateNotPassed(gate);
        }
    }
    ClaimReadiness::Ready
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_suit_state() -> SuitSafetyState {
        SuitSafetyState {
            suit_pressure_pa: 30_000.0,
            oxygen_partial_pressure_pa: 20_000.0,
            co2_partial_pressure_pa: 400.0,
            wearer_core_temperature_k: 310.0,
            assist_battery_soc: 0.8,
            pressure_integrity: 0.999,
            assist_electronics_health: 0.99,
        }
    }

    #[test]
    fn lunar_reference_requires_valid_caller_supplied_thermal_state() {
        assert!(SpaceEnvironmentState::lunar_surface(250.0, 0.1).is_valid());
        assert!(!SpaceEnvironmentState::lunar_surface(f64::NAN, 0.1).is_valid());
    }

    #[test]
    fn safety_kernel_clamps_but_does_not_invent_authority() {
        let kernel = SpaceExosuitSafetyKernel::new(CertifiedAssistEnvelope::simulation_reference());
        let env = SpaceEnvironmentState::lunar_surface(250.0, 0.1);
        let proposed = ExoskeletonCommand {
            joint_torques: [1.0, -1.0, 0.2, -0.2, 0.9, -0.9],
            stiffness_gain: 1.0,
            damping_gain: 1.0,
        };
        let decision = kernel.evaluate(&env, &nominal_suit_state(), &proposed);
        assert!(decision.permitted);
        assert!(decision.reasons.is_empty());
        assert_eq!(decision.command.joint_torques[0], 0.35);
        assert_eq!(decision.command.joint_torques[1], -0.35);
        assert_eq!(decision.command.joint_torques[2], 0.2);
        assert_eq!(decision.command.stiffness_gain, 0.5);
        assert_eq!(decision.command.damping_gain, 0.5);
    }

    #[test]
    fn pressure_integrity_failure_removes_powered_assist() {
        let kernel = SpaceExosuitSafetyKernel::new(CertifiedAssistEnvelope::simulation_reference());
        let env = SpaceEnvironmentState::lunar_surface(250.0, 0.1);
        let mut state = nominal_suit_state();
        state.pressure_integrity = 0.8;
        let proposed = ExoskeletonCommand {
            joint_torques: [0.2; NUM_ACTUATORS],
            stiffness_gain: 0.4,
            damping_gain: 0.4,
        };
        let decision = kernel.evaluate(&env, &state, &proposed);
        assert!(!decision.permitted);
        assert!(decision
            .reasons
            .contains(&AssistDenialReason::PressureIntegrityBelowFloor));
        assert_eq!(decision.command.joint_torques, [0.0; NUM_ACTUATORS]);
        assert_eq!(decision.command.stiffness_gain, 0.0);
    }

    #[test]
    fn non_finite_command_fails_closed() {
        let kernel = SpaceExosuitSafetyKernel::new(CertifiedAssistEnvelope::simulation_reference());
        let env = SpaceEnvironmentState::lunar_surface(250.0, 0.1);
        let mut proposed = ExoskeletonCommand::zero();
        proposed.joint_torques[3] = f32::NAN;
        let decision = kernel.evaluate(&env, &nominal_suit_state(), &proposed);
        assert!(!decision.permitted);
        assert_eq!(decision.reasons, vec![AssistDenialReason::InvalidCommand]);
        assert!(decision.command.joint_torques.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn simulation_evidence_cannot_support_qualification_claim() {
        let measurement = BenchmarkMeasurement {
            gate: SpaceExosuitBenchmarkGate::MobilityWorkload,
            protocol_id: "same-protocol-v1".into(),
            baseline_system: "baseline".into(),
            candidate_system: "candidate".into(),
            baseline_value: 100.0,
            candidate_value: 70.0,
            baseline_uncertainty: 2.0,
            candidate_uncertainty: 2.0,
            direction: MetricDirection::LowerIsBetter,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        assert_eq!(
            measurement.evaluate(ExosuitEvidenceLevel::Qualification, 0.1),
            BenchmarkOutcome::InsufficientEvidence
        );
    }

    #[test]
    fn benchmark_uses_conservative_uncertainty_bounds() {
        let measurement = BenchmarkMeasurement {
            gate: SpaceExosuitBenchmarkGate::Dexterity,
            protocol_id: "tool-board-v1".into(),
            baseline_system: "baseline".into(),
            candidate_system: "candidate".into(),
            baseline_value: 100.0,
            candidate_value: 85.0,
            baseline_uncertainty: 3.0,
            candidate_uncertainty: 3.0,
            direction: MetricDirection::LowerIsBetter,
            evidence: ExosuitEvidenceLevel::HumanInLoop,
        };
        assert!(matches!(
            measurement.evaluate(ExosuitEvidenceLevel::HumanInLoop, 0.05),
            BenchmarkOutcome::Pass { .. }
        ));
        assert!(matches!(
            measurement.evaluate(ExosuitEvidenceLevel::HumanInLoop, 0.10),
            BenchmarkOutcome::Fail { .. }
        ));
    }

    #[test]
    fn integrated_superiority_claim_requires_every_qualified_gate() {
        let one_gate = BenchmarkMeasurement {
            gate: SpaceExosuitBenchmarkGate::MobilityWorkload,
            protocol_id: "eva-v1".into(),
            baseline_system: "baseline".into(),
            candidate_system: "candidate".into(),
            baseline_value: 100.0,
            candidate_value: 50.0,
            baseline_uncertainty: 0.0,
            candidate_uncertainty: 0.0,
            direction: MetricDirection::LowerIsBetter,
            evidence: ExosuitEvidenceLevel::Qualification,
        };
        assert_eq!(
            superiority_claim_readiness(&[one_gate], 0.05),
            ClaimReadiness::MissingGate(SpaceExosuitBenchmarkGate::Dexterity)
        );
    }
}
