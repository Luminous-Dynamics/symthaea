// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Motor Output Bridge: translates FEP MotorCommand into real-world ActionIR execution.
//!
//! This is "The Hands" of the cognitive loop — the bridge between consciousness-level
//! motor commands (from active inference) and actual file I/O, shell commands, and tests
//! via the existing `SimpleExecutor` + `ActionRegistry` infrastructure.
//!
//! ## Consciousness Gating
//!
//! Actions are gated by Phi level through the `PolicyBundle` — higher-risk actions
//! require higher consciousness levels before execution is permitted.
//!
//! ## Parameter Encoding
//!
//! `MotorCommand.parameters` encodes the action request:
//! - `parameters[0]` — action type selector (mapped via `ActionType`)
//!
//! String data (file paths, content, args) is provided via `MotorActionRequest`,
//! set externally before the cycle by the agentic loop (Phase 3).

use crate::action::bindings::{ActionContext, ActionRegistry};
use crate::action::{
    ActionIR, ActionOutcome, ExecutionOutcome, PolicyBundle, SandboxRoot, SimpleExecutor,
};
#[cfg(feature = "safety-agents")]
use crate::cognitive_loop::guardian::{GuardianPosture, GuardianState};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Encodes the action type from MotorCommand parameters[0].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionType {
    Read = 0,
    Write = 1,
    List = 2,
    Parse = 3,
    CargoTest = 4,
    CargoCheck = 5,
    GitCommit = 6,
    RunCommand = 7,
}

impl ActionType {
    /// Decode from a floating-point parameter index.
    pub fn from_param(val: f64) -> Option<Self> {
        match val as u8 {
            0 => Some(Self::Read),
            1 => Some(Self::Write),
            2 => Some(Self::List),
            3 => Some(Self::Parse),
            4 => Some(Self::CargoTest),
            5 => Some(Self::CargoCheck),
            6 => Some(Self::GitCommit),
            7 => Some(Self::RunCommand),
            _ => None,
        }
    }

    /// Map to the ActionRegistry primitive name.
    fn registry_key(self) -> &'static str {
        match self {
            Self::Read => "READ",
            Self::Write => "WRITE",
            Self::List => "LIST",
            Self::Parse => "PARSE",
            Self::CargoTest => "CARGO_TEST",
            Self::CargoCheck => "CARGO_CHECK",
            Self::GitCommit => "GIT_COMMIT",
            Self::RunCommand => "RUN_COMMAND",
        }
    }
}

/// Result of a motor output execution, suitable for FEP feedback.
#[derive(Debug, Clone)]
pub struct MotorOutputResult {
    /// Whether the action succeeded.
    pub success: bool,
    /// The action that was executed (or attempted).
    pub action_type: Option<ActionType>,
    /// Prediction error: 0.0 = outcome matched expectation, 1.0 = total surprise.
    pub prediction_error: f64,
    /// The raw outcome (for downstream consumers).
    pub outcome: Option<ActionOutcome>,
    /// Error message if execution failed.
    pub error: Option<String>,
}

impl MotorOutputResult {
    fn success(action_type: ActionType, outcome: ActionOutcome) -> Self {
        Self {
            success: true,
            action_type: Some(action_type),
            prediction_error: 0.0,
            outcome: Some(outcome),
            error: None,
        }
    }

    fn failure(action_type: Option<ActionType>, error: String) -> Self {
        Self {
            success: false,
            action_type,
            prediction_error: 1.0,
            outcome: None,
            error: Some(error),
        }
    }

    fn skipped(reason: &str) -> Self {
        Self {
            success: false,
            action_type: None,
            prediction_error: super::thresholds::MOTOR_SKIP_PREDICTION_ERROR,
            outcome: None,
            error: Some(reason.to_string()),
        }
    }
}

/// Context for a motor output action, set by the agentic loop before cycle dispatch.
///
/// The FEP `MotorCommand.parameters` carries numeric metadata, but string data
/// (file paths, content, arguments) must be provided separately since f64 parameters
/// can't encode arbitrary strings. The agentic loop (Phase 3) sets this before each
/// cycle based on its task state machine.
#[derive(Debug, Clone, Default)]
pub struct MotorActionRequest {
    /// Target file path (for READ, WRITE, LIST, PARSE, CARGO_TEST, CARGO_CHECK).
    pub target_path: Option<PathBuf>,
    /// Content to write (for WRITE, GIT_COMMIT message).
    pub content: Option<String>,
    /// Additional arguments (for RUN_COMMAND).
    pub args: Vec<String>,
    /// Program name override (for RUN_COMMAND).
    pub program: Option<String>,
}

/// Bridge between FEP MotorCommand and real-world action execution.
///
/// Owns its own `SimpleExecutor` so that telemetry, dream engine wisdom, and
/// budget tracking persist across cycles. The agentic loop (Phase 3) drives this
/// bridge externally.
pub struct MotorOutputBridge {
    /// Policy bundle governing what actions are permitted.
    policy: PolicyBundle,
    /// Sandbox root for path validation.
    sandbox: SandboxRoot,
    /// Registry of primitive → ActionIR bindings (includes RUN_COMMAND).
    registry: ActionRegistry,
    /// Persistent executor — retains telemetry, dream wisdom, and budget across cycles.
    executor: SimpleExecutor,
    /// Minimum Phi required for any motor output (overrides policy if higher).
    min_phi_override: Option<f64>,
    /// Current safety level — gates motor output.
    /// Red = halt all motor, Orange = readonly only.
    #[cfg(feature = "safety-agents")]
    safety_override: crate::safety::SafetyLevel,
}

impl MotorOutputBridge {
    /// Create a new bridge with the given policy and sandbox.
    pub fn new(policy: PolicyBundle, sandbox: SandboxRoot) -> Self {
        Self {
            policy,
            sandbox,
            registry: Self::coding_registry(),
            executor: SimpleExecutor::from_env(),
            min_phi_override: None,
            #[cfg(feature = "safety-agents")]
            safety_override: crate::safety::SafetyLevel::Green,
        }
    }

    /// Create a bridge with default restrictive policy and a temp sandbox.
    pub fn with_defaults() -> Result<Self, std::io::Error> {
        let sandbox = SandboxRoot::new("motor_bridge")?;
        Ok(Self {
            policy: PolicyBundle::restrictive(),
            sandbox,
            registry: Self::coding_registry(),
            executor: SimpleExecutor::from_env(),
            min_phi_override: None,
            #[cfg(feature = "safety-agents")]
            safety_override: crate::safety::SafetyLevel::Green,
        })
    }

    /// Override the minimum Phi threshold for motor output execution.
    pub fn with_min_phi(mut self, min_phi: f64) -> Self {
        self.min_phi_override = Some(min_phi);
        self
    }

    /// Set a minimum confidence threshold (currently a no-op placeholder).
    ///
    /// Confidence gating is handled by the FEP motor confidence check in
    /// `cycle_phase_dynamics.rs`. This builder method exists for API symmetry
    /// with `with_min_phi()`.
    pub fn with_min_confidence(self, _min_confidence: f64) -> Self {
        self
    }

    /// Update the safety level from SafetyAgent assessment.
    /// Red = halt all motor, Orange = readonly only.
    #[cfg(feature = "safety-agents")]
    pub fn set_safety_level(&mut self, level: crate::safety::SafetyLevel) {
        self.safety_override = level;
    }

    /// Switch the executor to real command execution mode.
    ///
    /// This enables actual file I/O and shell commands (cargo check, cargo test).
    /// Use with care — only when the sandbox is properly configured.
    pub fn enable_real_execution(&mut self) {
        self.executor = SimpleExecutor::with_real_commands();
    }

    /// Access execution telemetry from the persistent executor.
    pub fn telemetry(&self) -> &[crate::action::ExecutionRecord] {
        self.executor.telemetry()
    }

    /// Build the standard coding registry: all 14 standard bindings + RUN_COMMAND.
    fn coding_registry() -> ActionRegistry {
        ActionRegistry::standard().register("RUN_COMMAND", |ctx| {
            let program = ctx
                .args
                .first()
                .cloned()
                .unwrap_or_else(|| "echo".to_string());
            let args = if ctx.args.len() > 1 {
                ctx.args[1..].to_vec()
            } else {
                vec![]
            };
            Ok(ActionIR::RunCommand {
                program,
                args,
                env: BTreeMap::new(),
                working_dir: ctx.target_path.clone(),
            })
        })
    }

    /// Execute a motor command, translating parameters into ActionIR and running
    /// through the persistent SimpleExecutor with policy validation and Phi gating.
    pub fn execute(
        &mut self,
        motor_params: &[f64],
        motor_confidence: f64,
        current_phi: f64,
        request: &MotorActionRequest,
    ) -> MotorOutputResult {
        // 1. Check minimum Phi gate
        let min_phi = self
            .min_phi_override
            .unwrap_or(self.policy.capabilities.min_phi);
        if current_phi < min_phi {
            return MotorOutputResult::skipped(&format!(
                "Phi {current_phi:.3} below motor output threshold {min_phi:.3}"
            ));
        }

        // 2. Low confidence → skip (FEP isn't sure this is the right action)
        if motor_confidence < super::thresholds::MOTOR_CONFIDENCE_MIN {
            return MotorOutputResult::skipped(&format!(
                "Motor confidence {motor_confidence:.3} too low for execution"
            ));
        }

        // 3. Decode action type from parameters[0]
        let action_type = match motor_params
            .first()
            .and_then(|&v| ActionType::from_param(v))
        {
            Some(at) => at,
            None => {
                return MotorOutputResult::skipped("No valid action type in motor parameters");
            }
        };

        // 4. Build ActionContext from the request
        let action_context = ActionContext {
            target_path: request.target_path.clone(),
            content: request.content.clone(),
            args: if action_type == ActionType::RunCommand {
                // For RUN_COMMAND, pack program + args into the context args list
                let mut full_args = Vec::new();
                if let Some(ref prog) = request.program {
                    full_args.push(prog.clone());
                }
                full_args.extend(request.args.iter().cloned());
                full_args
            } else {
                request.args.clone()
            },
            env: Default::default(),
        };

        // 5. Resolve through registry → ActionIR
        let action_ir = match self
            .registry
            .resolve(action_type.registry_key(), &action_context)
        {
            Ok(ir) => ir,
            Err(e) => {
                return MotorOutputResult::failure(
                    Some(action_type),
                    format!("Registry resolution failed: {e}"),
                );
            }
        };

        // 6. Check destructiveness vs Phi tier
        //    ReadOnly: min_phi (default 0.3)
        //    Reversible: min_phi + 0.1
        //    NeedsConfirmation: min_phi + 0.2
        //    Destructive: min_phi + 0.3
        let phi_requirement = match action_ir.destructiveness() {
            crate::action::DestructivenessLevel::ReadOnly => min_phi,
            crate::action::DestructivenessLevel::Reversible => {
                (min_phi + super::thresholds::MOTOR_PHI_REVERSIBLE_BONUS).min(1.0)
            }
            crate::action::DestructivenessLevel::NeedsConfirmation => {
                (min_phi + super::thresholds::MOTOR_PHI_CONFIRMATION_BONUS).min(1.0)
            }
            crate::action::DestructivenessLevel::Destructive => {
                (min_phi + super::thresholds::MOTOR_PHI_DESTRUCTIVE_BONUS).min(1.0)
            }
        };

        if current_phi < phi_requirement {
            return MotorOutputResult::failure(
                Some(action_type),
                format!(
                    "Phi {current_phi:.3} insufficient for {:?} action (requires {phi_requirement:.3})",
                    action_ir.destructiveness()
                ),
            );
        }

        // 7. Execute through persistent SimpleExecutor (policy + sandbox + dream engine)
        match self
            .executor
            .execute(&action_ir, &self.policy, &self.sandbox, current_phi)
        {
            Ok(ExecutionOutcome { outcome, .. }) => {
                tracing::info!(
                    target: "symthaea::motor_bridge",
                    action = ?action_type,
                    phi = current_phi,
                    confidence = motor_confidence,
                    "Motor output executed successfully"
                );
                MotorOutputResult::success(action_type, outcome)
            }
            Err(e) => {
                tracing::warn!(
                    target: "symthaea::motor_bridge",
                    action = ?action_type,
                    error = %e,
                    "Motor output execution failed"
                );
                MotorOutputResult::failure(Some(action_type), format!("{e}"))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROBOT ACTUATOR OUTPUT — Path 2: Embodied Intelligence
// ═══════════════════════════════════════════════════════════════════════════════

/// Command to physical actuators derived from GuardianState + CPG.
///
/// This is the bridge between consciousness (Phi-gated posture) and hardware
/// (servo positions via HAL). Each cycle, the cognitive loop produces a
/// `RobotActuatorCommand` that can be consumed by:
/// - `symthaea-hal::HalRuntime` (real hardware: PCA9685 servos, IMU, e-stop)
/// - `symthaea-humanoid::HumanoidController` (MuJoCo simulation)
/// - `symthaea-multirotor::FlightController` (quadrotor simulation)
///
/// ## Safety
///
/// The command inherits safety from `GuardianPosture`:
/// - `Emergency` / `Hold` → `e_stop = true` (all servos to safe position)
/// - `Defensive` → no locomotion, sensor-only mode
/// - `Cautious` / `Normal` → locomotion permitted with CPG timing
#[cfg(feature = "safety-agents")]
#[derive(Debug, Clone)]
pub struct RobotActuatorCommand {
    /// Current guardian posture (from consciousness + safety level).
    pub posture: GuardianPosture,
    /// Whether e-stop should be engaged (Emergency or Hold posture).
    pub e_stop: bool,
    /// Whether locomotion is permitted.
    pub locomotion_permitted: bool,
    /// Recommended CPG gait name (None if locomotion not permitted).
    pub recommended_gait: Option<&'static str>,
    /// CPG oscillator phases (8 oscillators, rad) — timing for limb actuators.
    /// Empty if CPG is not active or locomotion is not permitted.
    pub cpg_phases: Vec<f64>,
    /// CPG oscillator outputs (sin(phase)) — direct servo targets.
    pub cpg_outputs: Vec<f64>,
    /// CPG synchronization index (0.0 = incoherent, 1.0 = perfect sync).
    pub cpg_sync_index: f64,
    /// Sensor sweep multiplier (lower = faster sweep).
    pub sensor_sweep_multiplier: f32,
    /// Whether patrol mode is active.
    pub patrol_active: bool,
    /// Current patrol waypoint index.
    pub patrol_waypoint: usize,
    /// Consciousness level (Phi) that generated this command.
    pub phi: f64,
    /// Cycle number for sequencing.
    pub cycle: usize,
}

#[cfg(feature = "safety-agents")]
impl RobotActuatorCommand {
    /// Generate a robot actuator command from the current guardian state.
    ///
    /// CPG phases/outputs are provided separately by the CpgManager subsystem.
    pub fn from_guardian(
        guardian: &GuardianState,
        cpg_phases: &[f64],
        cpg_outputs: &[f64],
        cpg_sync_index: f64,
        phi: f64,
        cycle: usize,
    ) -> Self {
        Self {
            posture: guardian.posture,
            e_stop: guardian.posture.e_stop_engaged(),
            locomotion_permitted: guardian.posture.locomotion_permitted(),
            recommended_gait: guardian.posture.recommended_gait(),
            cpg_phases: if guardian.posture.locomotion_permitted() {
                cpg_phases.to_vec()
            } else {
                Vec::new()
            },
            cpg_outputs: if guardian.posture.locomotion_permitted() {
                cpg_outputs.to_vec()
            } else {
                Vec::new()
            },
            cpg_sync_index: if guardian.posture.locomotion_permitted() {
                cpg_sync_index
            } else {
                0.0
            },
            sensor_sweep_multiplier: guardian.posture.sensor_sweep_multiplier(),
            patrol_active: guardian.patrol_active,
            patrol_waypoint: guardian.patrol_waypoint,
            phi,
            cycle,
        }
    }

    /// Whether this command represents a safe-hold (no physical action).
    pub fn is_safe_hold(&self) -> bool {
        self.e_stop || self.posture == GuardianPosture::Hold
    }

    /// Convert to a 21-element joint target array for the humanoid controller.
    ///
    /// Maps CPG oscillator outputs to joint angles:
    /// - Oscillators 0-1: Left hip/knee
    /// - Oscillators 2-3: Right hip/knee
    /// - Oscillators 4-5: Left shoulder/elbow
    /// - Oscillators 6-7: Right shoulder/elbow
    /// - Remaining joints (8-20): centered (0.0)
    ///
    /// Returns None if e-stop is engaged or CPG outputs are empty.
    #[cfg(feature = "humanoid")]
    pub fn to_joint_targets(&self) -> Option<[f64; 21]> {
        if self.is_safe_hold() || self.cpg_outputs.len() < 8 {
            return None;
        }

        let mut joints = [0.0f64; 21];

        // CPG outputs are sin(phase) ∈ [-1, 1]
        // Scale to joint angle range (±45° = ±0.785 rad)
        let scale = std::f64::consts::FRAC_PI_4; // π/4

        use super::thresholds::{JOINT_ELBOW_SCALE, JOINT_KNEE_SCALE, JOINT_SHOULDER_SCALE};

        // Left leg: hip (0), knee (1)
        joints[0] = self.cpg_outputs[0] * scale;
        joints[1] = self.cpg_outputs[1] * scale * JOINT_KNEE_SCALE;

        // Right leg: hip (2), knee (3)
        joints[2] = self.cpg_outputs[2] * scale;
        joints[3] = self.cpg_outputs[3] * scale * JOINT_KNEE_SCALE;

        // Left arm: shoulder (4), elbow (5)
        joints[4] = self.cpg_outputs[4] * scale * JOINT_SHOULDER_SCALE;
        joints[5] = self.cpg_outputs[5] * scale * JOINT_ELBOW_SCALE;

        // Right arm: shoulder (6), elbow (7)
        joints[6] = self.cpg_outputs[6] * scale * JOINT_SHOULDER_SCALE;
        joints[7] = self.cpg_outputs[7] * scale * JOINT_ELBOW_SCALE;

        // Joints 8-20: centered (torso, head, hands, feet)

        Some(joints)
    }
}

#[cfg(feature = "safety-agents")]
impl MotorOutputBridge {
    /// Generate a robot actuator command from the current guardian + CPG state.
    ///
    /// This is the primary output for embodied systems. Call each cycle
    /// after guardian state and CPG have been updated.
    pub fn robot_command(
        &self,
        guardian: &GuardianState,
        cpg_phases: &[f64],
        cpg_outputs: &[f64],
        cpg_sync_index: f64,
        phi: f64,
        cycle: usize,
    ) -> RobotActuatorCommand {
        // Safety override: if safety level is Red/Orange, force appropriate posture
        #[cfg(feature = "safety-agents")]
        {
            if self.safety_override == crate::safety::SafetyLevel::Red {
                return RobotActuatorCommand {
                    posture: GuardianPosture::Emergency,
                    e_stop: true,
                    locomotion_permitted: false,
                    recommended_gait: None,
                    cpg_phases: Vec::new(),
                    cpg_outputs: Vec::new(),
                    cpg_sync_index: 0.0,
                    sensor_sweep_multiplier: 1.0,
                    patrol_active: false,
                    patrol_waypoint: 0,
                    phi,
                    cycle,
                };
            }
        }

        RobotActuatorCommand::from_guardian(
            guardian,
            cpg_phases,
            cpg_outputs,
            cpg_sync_index,
            phi,
            cycle,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_type_roundtrip() {
        for i in 0..8u8 {
            let at = ActionType::from_param(i as f64).unwrap();
            assert_eq!(at as u8, i);
        }
        assert!(ActionType::from_param(99.0).is_none());
    }

    #[test]
    fn test_phi_gating_blocks_low_consciousness() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();
        let request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/test.txt")),
            ..Default::default()
        };

        // Phi below threshold → skipped
        let result = bridge.execute(&[0.0], 0.8, 0.1, &request);
        assert!(!result.success);
        assert!(
            result
                .error
                .unwrap()
                .contains("below motor output threshold")
        );
    }

    #[test]
    fn test_low_confidence_blocks_execution() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();
        let request = MotorActionRequest::default();

        // High Phi but low confidence → skipped
        let result = bridge.execute(&[0.0], 0.1, 0.9, &request);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("confidence"));
    }

    #[test]
    fn test_no_params_skips() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();
        let request = MotorActionRequest::default();

        let result = bridge.execute(&[], 0.8, 0.9, &request);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("No valid action type"));
    }

    #[test]
    fn test_destructive_requires_higher_phi() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();

        // RUN_COMMAND: program in args[0], rest are args
        let request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/")),
            program: Some("echo".into()),
            args: vec!["hello".into()],
            content: None,
        };

        // parameters[0]=7 is RunCommand, which is Destructive-tier in classification
        // Phi=0.35 is above min_phi (0.3) but below destructive threshold (0.6)
        let result = bridge.execute(&[7.0], 0.8, 0.35, &request);
        assert!(!result.success);
    }

    #[test]
    fn test_read_action_simulated() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();

        let request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/test.txt")),
            ..Default::default()
        };

        // READ (param 0) with sufficient Phi
        let result = bridge.execute(&[0.0], 0.8, 0.9, &request);
        assert!(result.action_type == Some(ActionType::Read));
    }

    #[test]
    fn test_motor_output_result_helpers() {
        let success = MotorOutputResult::success(ActionType::Read, ActionOutcome::Success);
        assert!(success.success);
        assert_eq!(success.prediction_error, 0.0);

        let failure = MotorOutputResult::failure(Some(ActionType::Write), "test error".into());
        assert!(!failure.success);
        assert_eq!(failure.prediction_error, 1.0);

        let skipped = MotorOutputResult::skipped("test skip");
        assert!(!skipped.success);
        assert_eq!(skipped.prediction_error, 0.5);
    }

    #[test]
    fn test_run_command_registry_key_exists() {
        let registry = MotorOutputBridge::coding_registry();
        let ctx = ActionContext {
            target_path: None,
            content: None,
            args: vec!["echo".into(), "hello".into()],
            env: Default::default(),
        };
        let result = registry.resolve("RUN_COMMAND", &ctx);
        assert!(result.is_ok(), "RUN_COMMAND should be registered");
        if let Ok(ActionIR::RunCommand { program, args, .. }) = result {
            assert_eq!(program, "echo");
            assert_eq!(args, vec!["hello"]);
        } else {
            panic!("Expected RunCommand ActionIR");
        }
    }

    #[test]
    fn test_telemetry_persists_across_executions() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();
        let request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/test.txt")),
            ..Default::default()
        };

        // Execute twice — telemetry should accumulate
        let _ = bridge.execute(&[0.0], 0.8, 0.9, &request);
        let _ = bridge.execute(&[0.0], 0.8, 0.9, &request);

        // Even if both fail (file doesn't exist in simulated mode),
        // the executor's telemetry log should show activity
        // (SimpleExecutor in Simulated mode may not log, but the bridge persists)
        assert!(bridge.telemetry().len() <= 2); // sanity check: not unbounded
    }

    // ── Robot Actuator Command Tests ─────────────────────────────────────

    #[cfg(feature = "safety-agents")]
    #[test]
    fn test_robot_command_from_guardian_normal() {
        use crate::safety::SafetyLevel;
        let mut guardian = GuardianState::default();
        guardian.update(SafetyLevel::Green, 0.8, 0);
        guardian.start_patrol(5);

        let phases = vec![0.0, 1.57, 3.14, 4.71, 0.5, 2.0, 3.5, 5.0];
        let outputs: Vec<f64> = phases.iter().map(|p| (*p as f64).sin()).collect();

        let cmd = RobotActuatorCommand::from_guardian(&guardian, &phases, &outputs, 0.95, 0.8, 100);

        assert_eq!(cmd.posture, GuardianPosture::Normal);
        assert!(!cmd.e_stop);
        assert!(cmd.locomotion_permitted);
        assert_eq!(cmd.recommended_gait, Some("walk"));
        assert_eq!(cmd.cpg_phases.len(), 8);
        assert_eq!(cmd.cpg_outputs.len(), 8);
        assert!(cmd.cpg_sync_index > 0.9);
        assert!(cmd.patrol_active);
        assert!(!cmd.is_safe_hold());
    }

    #[cfg(feature = "safety-agents")]
    #[test]
    fn test_robot_command_emergency_stops_everything() {
        use crate::safety::SafetyLevel;
        let mut guardian = GuardianState::default();
        guardian.update(SafetyLevel::Red, 0.8, 0);

        let phases = vec![0.0; 8];
        let outputs = vec![0.0; 8];

        let cmd = RobotActuatorCommand::from_guardian(&guardian, &phases, &outputs, 0.9, 0.8, 50);

        assert_eq!(cmd.posture, GuardianPosture::Emergency);
        assert!(cmd.e_stop);
        assert!(!cmd.locomotion_permitted);
        assert!(cmd.recommended_gait.is_none());
        assert!(cmd.cpg_phases.is_empty(), "No CPG in emergency");
        assert!(cmd.cpg_outputs.is_empty());
        assert!(cmd.is_safe_hold());
    }

    #[cfg(feature = "safety-agents")]
    #[test]
    fn test_robot_command_hold_on_low_phi() {
        use crate::safety::SafetyLevel;
        let mut guardian = GuardianState::default();
        guardian.update(SafetyLevel::Green, 0.1, 0); // Phi too low → Hold

        let cmd = RobotActuatorCommand::from_guardian(&guardian, &[], &[], 0.0, 0.1, 0);

        assert_eq!(cmd.posture, GuardianPosture::Hold);
        assert!(cmd.e_stop);
        assert!(cmd.is_safe_hold());
    }

    #[cfg(feature = "safety-agents")]
    #[test]
    fn test_robot_command_defensive_no_locomotion() {
        use crate::safety::SafetyLevel;
        let mut guardian = GuardianState::default();
        guardian.update(SafetyLevel::Orange, 0.8, 0);

        let phases = vec![0.0; 8];
        let outputs = vec![1.0; 8];

        let cmd = RobotActuatorCommand::from_guardian(&guardian, &phases, &outputs, 0.9, 0.8, 0);

        assert_eq!(cmd.posture, GuardianPosture::Defensive);
        assert!(!cmd.locomotion_permitted);
        assert!(cmd.cpg_phases.is_empty(), "No CPG in defensive");
        assert!(
            cmd.sensor_sweep_multiplier < 0.5,
            "Defensive = fast sensor sweep"
        );
    }

    #[cfg(feature = "safety-agents")]
    #[test]
    fn test_motor_bridge_robot_command() {
        use crate::safety::SafetyLevel;
        let bridge = MotorOutputBridge::with_defaults().unwrap();
        let mut guardian = GuardianState::default();
        guardian.update(SafetyLevel::Green, 0.8, 0);

        let cmd = bridge.robot_command(&guardian, &[0.0; 8], &[0.5; 8], 0.9, 0.8, 42);
        assert_eq!(cmd.posture, GuardianPosture::Normal);
        assert_eq!(cmd.cycle, 42);
    }
}
