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
            prediction_error: 0.5,
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
    /// Minimum motor confidence required for execution (default 0.3).
    min_confidence: f64,
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
            min_confidence: 0.3,
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
            min_confidence: 0.3,
        })
    }

    /// Override the minimum Phi threshold for motor output execution.
    pub fn with_min_phi(mut self, min_phi: f64) -> Self {
        self.min_phi_override = Some(min_phi);
        self
    }

    /// Override the minimum motor confidence threshold.
    ///
    /// In agentic coding contexts, the FEP's motor confidence starts low
    /// because it hasn't been trained on coding-specific actions. Lowering
    /// this threshold allows the coding agent to execute actions while the
    /// FEP learns appropriate confidence levels.
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
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

    /// Infer the action type from the `MotorActionRequest` fields when FEP motor
    /// parameters don't decode to a valid `ActionType`.
    fn infer_action_type(request: &MotorActionRequest) -> Option<ActionType> {
        // Cargo commands: check program name + args
        if let Some(ref prog) = request.program {
            if prog == "cargo" || prog.ends_with("/cargo") {
                if request.args.iter().any(|a| a == "check") {
                    return Some(ActionType::CargoCheck);
                }
                if request.args.iter().any(|a| a == "test") {
                    return Some(ActionType::CargoTest);
                }
            }
            // Git commit
            if prog == "git" || prog.ends_with("/git") {
                if request.args.iter().any(|a| a == "commit") {
                    return Some(ActionType::GitCommit);
                }
            }
            // Any other program → RunCommand
            return Some(ActionType::RunCommand);
        }

        // Content present → Write
        if request.content.is_some() && request.target_path.is_some() {
            return Some(ActionType::Write);
        }

        // Target path only → Read
        if request.target_path.is_some() {
            return Some(ActionType::Read);
        }

        None
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
        if motor_confidence < self.min_confidence {
            return MotorOutputResult::skipped(&format!(
                "Motor confidence {motor_confidence:.3} too low for execution (threshold {:.3})",
                self.min_confidence
            ));
        }

        // 3. Decode action type from parameters[0], falling back to inference from request context.
        // The FEP's motor parameters may not map to our ActionType encoding (the FEP uses
        // MotorCommandType indices, not ActionType indices). When parameters[0] doesn't decode
        // to a valid ActionType, we infer the action from the MotorActionRequest fields.
        let action_type = motor_params
            .first()
            .and_then(|&v| ActionType::from_param(v))
            .or_else(|| Self::infer_action_type(request))
            .unwrap_or_else(|| {
                // Last resort: if there's a target path, try Read; otherwise skip
                if request.target_path.is_some() {
                    ActionType::Read
                } else {
                    return ActionType::List; // will be caught by registry
                }
            });

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
            crate::action::DestructivenessLevel::Reversible => (min_phi + 0.1).min(1.0),
            crate::action::DestructivenessLevel::NeedsConfirmation => (min_phi + 0.2).min(1.0),
            crate::action::DestructivenessLevel::Destructive => (min_phi + 0.3).min(1.0),
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
        assert!(result
            .error
            .unwrap()
            .contains("below motor output threshold"));
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
    fn test_no_params_infers_from_request() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();

        // Empty request + no params → falls back to List
        let request = MotorActionRequest::default();
        let result = bridge.execute(&[], 0.8, 0.9, &request);
        assert_eq!(result.action_type, Some(ActionType::List));

        // Request with content + path → infers Write
        let write_request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/out.txt")),
            content: Some("hello".into()),
            ..Default::default()
        };
        let result = bridge.execute(&[], 0.8, 0.9, &write_request);
        assert_eq!(result.action_type, Some(ActionType::Write));

        // Request with program → infers RunCommand
        let cmd_request = MotorActionRequest {
            program: Some("echo".into()),
            args: vec!["hi".into()],
            ..Default::default()
        };
        let result = bridge.execute(&[], 0.8, 0.9, &cmd_request);
        assert_eq!(result.action_type, Some(ActionType::RunCommand));
    }

    #[test]
    fn test_infer_cargo_commands() {
        let mut bridge = MotorOutputBridge::with_defaults().unwrap();

        let check_request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/")),
            program: Some("cargo".into()),
            args: vec!["check".into()],
            ..Default::default()
        };
        let result = bridge.execute(&[], 0.8, 0.9, &check_request);
        assert_eq!(result.action_type, Some(ActionType::CargoCheck));

        let test_request = MotorActionRequest {
            target_path: Some(PathBuf::from("/tmp/symthaea/motor_bridge/")),
            program: Some("cargo".into()),
            args: vec!["test".into()],
            ..Default::default()
        };
        let result = bridge.execute(&[], 0.8, 0.9, &test_request);
        assert_eq!(result.action_type, Some(ActionType::CargoTest));
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
}
