// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autonomous Lab Controller — Genesis Mission Challenge 10
//!
//! A generic closed-loop controller for simulated or supervised wet-lab workflows.
//! Uses deterministic action selection and ethics gating to evaluate multi-step
//! protocols (e.g., iPSC reprogramming, gene editing, tissue culture).
//!
//! This crate does **not** certify unattended physical execution. Production
//! instrument adapters must add hardware interlocks, calibrated/timestamped
//! readings, authorization, idempotent commands, and emergency-stop handling.
//!
//! The controller is generic over `InstrumentAdapter`, allowing any lab
//! instrument (incubator, sequencer, microscope) to be plugged in.

use serde::{Deserialize, Serialize};

use crate::ethics_gate::EthicsGate;
use crate::fep_agent::{CultureAction, CultureFepAgent};
use crate::multi_scale_predictor::MultiScalePredictor;
use crate::types::{CellState, CultureEnvironment};

// ── Instrument Adapter Trait ───────────────────────────────────────────────

/// Generic lab instrument protocol adapter.
///
/// Implement this trait to connect the autonomous controller to any
/// lab instrument (incubator, bioreactor, sequencer, microscope, etc.).
pub trait InstrumentAdapter: Send {
    /// Read the current cell state from the instrument.
    fn read_cell_state(&self) -> CellState;

    /// Read the current environment from the instrument.
    fn read_environment(&self) -> CultureEnvironment;

    /// Execute a culture action on the instrument.
    ///
    /// Returns Ok(()) if the action was executed successfully,
    /// or Err with an explanation if it failed.
    fn execute_action(&mut self, action: CultureAction) -> Result<(), String>;

    /// Name of this instrument (for logging).
    fn name(&self) -> &str;
}

// ── Mock Instrument ────────────────────────────────────────────────────────

/// Mock lab instrument for testing. Tracks executed actions.
pub struct MockInstrument {
    cell_state: CellState,
    environment: CultureEnvironment,
    executed_actions: Vec<CultureAction>,
    fail_on: Option<CultureAction>,
}

impl MockInstrument {
    /// Create a mock instrument with default somatic cell and standard culture environment.
    pub fn new() -> Self {
        Self {
            cell_state: CellState::new_somatic(),
            environment: CultureEnvironment::standard(),
            executed_actions: Vec::new(),
            fail_on: None,
        }
    }

    /// Create a mock with specific initial conditions.
    pub fn with_state(cell_state: CellState, environment: CultureEnvironment) -> Self {
        Self {
            cell_state,
            environment,
            executed_actions: Vec::new(),
            fail_on: None,
        }
    }

    /// Cause a specific action to fail when attempted.
    pub fn set_fail_on(&mut self, action: CultureAction) {
        self.fail_on = Some(action);
    }

    /// All actions that were executed.
    pub fn executed_actions(&self) -> &[CultureAction] {
        &self.executed_actions
    }

    /// Mutate the environment for testing.
    pub fn set_environment(&mut self, env: CultureEnvironment) {
        self.environment = env;
    }

    /// Mutate the cell state for testing.
    pub fn set_cell_state(&mut self, state: CellState) {
        self.cell_state = state;
    }
}

impl Default for MockInstrument {
    fn default() -> Self {
        Self::new()
    }
}

impl InstrumentAdapter for MockInstrument {
    fn read_cell_state(&self) -> CellState {
        self.cell_state.clone()
    }

    fn read_environment(&self) -> CultureEnvironment {
        self.environment.clone()
    }

    fn execute_action(&mut self, action: CultureAction) -> Result<(), String> {
        if self.fail_on == Some(action) {
            return Err(format!("Mock failure on {:?}", action));
        }
        self.executed_actions.push(action);
        Ok(())
    }

    fn name(&self) -> &str {
        "MockInstrument"
    }
}

// ── Lab Protocol ───────────────────────────────────────────────────────────

/// A single step in a lab protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStep {
    /// Human-readable step name.
    pub name: String,
    /// Duration of this step in seconds.
    pub duration_seconds: f64,
    /// Minimum viability required to proceed.
    pub min_viability: f64,
    /// Expected action from the FEP agent.
    pub expected_action: Option<CultureAction>,
    /// Whether this step requires ethics gate approval.
    pub requires_ethics_check: bool,
}

/// A complete lab protocol (sequence of steps with success criteria).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabProtocol {
    /// Protocol name.
    pub name: String,
    /// Steps to execute in order.
    pub steps: Vec<ProtocolStep>,
    /// Global minimum viability threshold (below = abort).
    pub safety_viability_floor: f64,
    /// Maximum total duration in seconds.
    pub max_duration_seconds: f64,
}

impl LabProtocol {
    /// Standard iPSC reprogramming protocol (4 steps).
    pub fn ipsc_reprogramming() -> Self {
        Self {
            name: "iPSC Reprogramming".to_string(),
            steps: vec![
                ProtocolStep {
                    name: "Factor Delivery".to_string(),
                    duration_seconds: 86_400.0, // 1 day
                    min_viability: 0.8,
                    expected_action: None,
                    requires_ethics_check: true,
                },
                ProtocolStep {
                    name: "Early Response (MET)".to_string(),
                    duration_seconds: 259_200.0, // 3 days
                    min_viability: 0.6,
                    expected_action: Some(CultureAction::AdjustTemp),
                    requires_ethics_check: false,
                },
                ProtocolStep {
                    name: "Epigenetic Remodeling".to_string(),
                    duration_seconds: 604_800.0, // 7 days
                    min_viability: 0.5,
                    expected_action: None,
                    requires_ethics_check: false,
                },
                ProtocolStep {
                    name: "Colony Stabilization".to_string(),
                    duration_seconds: 604_800.0, // 7 days
                    min_viability: 0.7,
                    expected_action: Some(CultureAction::PassageCells),
                    requires_ethics_check: false,
                },
            ],
            safety_viability_floor: 0.2,
            max_duration_seconds: 2_419_200.0, // 28 days
        }
    }
}

// ── Lab Controller ─────────────────────────────────────────────────────────

/// Result of a single controller step.
#[derive(Debug, Clone)]
/// Result of executing a single protocol step.
pub struct LabControllerStepResult {
    /// Current step index.
    pub step_index: usize,
    /// Step name.
    pub step_name: String,
    /// Action selected by FEP agent.
    pub selected_action: CultureAction,
    /// Whether the action was executed.
    pub action_executed: bool,
    /// Failure reason (if any).
    pub failure_reason: Option<String>,
    /// Predicted state at the step's horizon.
    pub predicted_state_similarity: f32,
    /// Action declared by the protocol author, when one was provided.
    pub expected_action: Option<CultureAction>,
    /// Whether the selected action matched the declared expectation.
    /// `None` means no expectation was declared or execution was blocked before
    /// action selection.
    pub action_matches_expectation: Option<bool>,
    /// Current cell viability.
    pub viability: f64,
}

/// Outcome of running a complete [`LabProtocol`] through the controller.
#[derive(Debug, Clone)]
pub struct ProtocolResult {
    /// Whether the protocol completed successfully.
    pub success: bool,
    /// Step results for each step.
    pub step_results: Vec<LabControllerStepResult>,
    /// Why the protocol failed (if applicable).
    pub failure_reason: Option<String>,
    /// Total steps completed.
    pub steps_completed: usize,
}

/// Generic closed-loop lab controller.
///
/// Combines an FEP agent, multi-scale predictor, and ethics gate
/// to autonomously execute lab protocols on any instrument.
///
/// # Genesis Mission Challenge 10
///
/// Addresses the DOE Autonomous Experimentation challenge by providing
/// a consciousness-inspired controller that can safely run multi-step
/// protocols with real-time monitoring and FEP-based decision-making.
pub struct LabController<I: InstrumentAdapter> {
    instrument: I,
    agent: CultureFepAgent,
    predictor: MultiScalePredictor,
    ethics: EthicsGate,
}

impl<I: InstrumentAdapter> LabController<I> {
    /// Create a new controller with the given instrument.
    pub fn new(instrument: I, ethics: EthicsGate) -> Self {
        Self {
            instrument,
            agent: CultureFepAgent::new(),
            predictor: MultiScalePredictor::new(),
            ethics,
        }
    }

    /// Run a single step of a protocol.
    ///
    /// 1. Read current state from instrument
    /// 2. Check viability floor
    /// 3. Check ethics gate (if required)
    /// 4. Predict future state
    /// 5. Select action via FEP
    /// 6. Execute action on instrument
    pub fn execute_step(
        &mut self,
        step: &ProtocolStep,
        step_index: usize,
    ) -> LabControllerStepResult {
        let cell_state = self.instrument.read_cell_state();
        let environment = self.instrument.read_environment();

        if !step.duration_seconds.is_finite() || step.duration_seconds < 0.0 {
            return LabControllerStepResult {
                step_index,
                step_name: step.name.clone(),
                selected_action: CultureAction::AbortProtocol,
                action_executed: false,
                failure_reason: Some(format!(
                    "Invalid step duration {}; expected a finite non-negative value",
                    step.duration_seconds
                )),
                predicted_state_similarity: 0.0,
                expected_action: step.expected_action,
                action_matches_expectation: None,
                viability: cell_state.viability,
            };
        }
        if !step.min_viability.is_finite() || !(0.0..=1.0).contains(&step.min_viability) {
            return LabControllerStepResult {
                step_index,
                step_name: step.name.clone(),
                selected_action: CultureAction::AbortProtocol,
                action_executed: false,
                failure_reason: Some(format!(
                    "Invalid minimum viability {}; expected [0, 1]",
                    step.min_viability
                )),
                predicted_state_similarity: 0.0,
                expected_action: step.expected_action,
                action_matches_expectation: None,
                viability: cell_state.viability,
            };
        }

        // Check ethics gate
        if step.requires_ethics_check
            && let Err(reason) = self.ethics.check_manipulation(&step.name)
        {
            return LabControllerStepResult {
                step_index,
                step_name: step.name.clone(),
                selected_action: CultureAction::AbortProtocol,
                action_executed: false,
                failure_reason: Some(format!("Ethics gate: {}", reason)),
                predicted_state_similarity: 0.0,
                expected_action: step.expected_action,
                action_matches_expectation: None,
                viability: cell_state.viability,
            };
        }

        // Check viability floor
        if cell_state.viability < step.min_viability {
            return LabControllerStepResult {
                step_index,
                step_name: step.name.clone(),
                selected_action: CultureAction::AbortProtocol,
                action_executed: false,
                failure_reason: Some(format!(
                    "Viability {:.3} below step minimum {:.3}",
                    cell_state.viability, step.min_viability
                )),
                predicted_state_similarity: 0.0,
                expected_action: step.expected_action,
                action_matches_expectation: None,
                viability: cell_state.viability,
            };
        }

        // Predict future state
        let encoded = crate::cell_encoder::encode_cell_state(&cell_state);
        self.predictor.observe(&encoded, 1.0);
        let predicted = self
            .predictor
            .predict_at_horizon(&encoded, step.duration_seconds as f32);
        let predicted_sim = predicted.similarity(&encoded);

        // FEP action selection
        let action = self.agent.select_action(&cell_state, &environment);

        // Execute action
        let (executed, failure) = match self.instrument.execute_action(action) {
            Ok(()) => (true, None),
            Err(e) => (false, Some(e)),
        };

        LabControllerStepResult {
            step_index,
            step_name: step.name.clone(),
            selected_action: action,
            action_executed: executed,
            failure_reason: failure,
            predicted_state_similarity: predicted_sim,
            expected_action: step.expected_action,
            action_matches_expectation: step.expected_action.map(|expected| expected == action),
            viability: cell_state.viability,
        }
    }

    /// Run a complete protocol.
    ///
    /// Enforces protocol-level `safety_viability_floor` and `max_duration_seconds`
    /// in addition to per-step viability checks.
    pub fn run_protocol(&mut self, protocol: &LabProtocol) -> ProtocolResult {
        let mut step_results = Vec::new();
        let mut elapsed_seconds = 0.0f64;

        if !protocol.max_duration_seconds.is_finite() || protocol.max_duration_seconds < 0.0 {
            return ProtocolResult {
                success: false,
                failure_reason: Some(format!(
                    "Invalid protocol max duration {}; expected zero (unlimited) or a finite positive value",
                    protocol.max_duration_seconds
                )),
                steps_completed: 0,
                step_results,
            };
        }
        if !protocol.safety_viability_floor.is_finite()
            || !(0.0..=1.0).contains(&protocol.safety_viability_floor)
        {
            return ProtocolResult {
                success: false,
                failure_reason: Some(format!(
                    "Invalid protocol safety viability floor {}; expected [0, 1]",
                    protocol.safety_viability_floor
                )),
                steps_completed: 0,
                step_results,
            };
        }

        for (i, step) in protocol.steps.iter().enumerate() {
            // Preflight the entire next step so the controller never executes an
            // action that would make the protocol exceed its declared limit.
            let next_elapsed = elapsed_seconds + step.duration_seconds;
            if protocol.max_duration_seconds > 0.0
                && (!next_elapsed.is_finite() || next_elapsed > protocol.max_duration_seconds)
            {
                return ProtocolResult {
                    success: false,
                    failure_reason: Some(format!(
                        "Step {} would exceed max duration {:.0}s (projected {:.0}s)",
                        i, protocol.max_duration_seconds, next_elapsed
                    )),
                    steps_completed: i,
                    step_results,
                };
            }

            let result = self.execute_step(step, i);
            let failed = result.failure_reason.is_some();

            // Check protocol-level viability floor (only if step itself didn't fail)
            if !failed && result.viability < protocol.safety_viability_floor {
                let reason = format!(
                    "Viability {:.3} below protocol safety floor {:.3}",
                    result.viability, protocol.safety_viability_floor
                );
                step_results.push(result);
                return ProtocolResult {
                    success: false,
                    failure_reason: Some(reason),
                    steps_completed: i,
                    step_results,
                };
            }

            elapsed_seconds = next_elapsed;
            step_results.push(result);

            if failed {
                return ProtocolResult {
                    success: false,
                    failure_reason: step_results.last().and_then(|r| r.failure_reason.clone()),
                    steps_completed: i,
                    step_results,
                };
            }
        }

        ProtocolResult {
            success: true,
            failure_reason: None,
            steps_completed: protocol.steps.len(),
            step_results,
        }
    }

    /// Access the instrument.
    pub fn instrument(&self) -> &I {
        &self.instrument
    }

    /// Mutable access to the instrument.
    pub fn instrument_mut(&mut self) -> &mut I {
        &mut self.instrument
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_instrument_default() {
        let mock = MockInstrument::new();
        let cell = mock.read_cell_state();
        assert!(cell.viability > 0.9);
        assert!(mock.executed_actions().is_empty());
    }

    #[test]
    fn test_mock_instrument_execute_action() {
        let mut mock = MockInstrument::new();
        assert!(mock.execute_action(CultureAction::AdjustTemp).is_ok());
        assert_eq!(mock.executed_actions().len(), 1);
        assert_eq!(mock.executed_actions()[0], CultureAction::AdjustTemp);
    }

    #[test]
    fn test_mock_instrument_fail_on() {
        let mut mock = MockInstrument::new();
        mock.set_fail_on(CultureAction::AdjustTemp);
        assert!(mock.execute_action(CultureAction::AdjustTemp).is_err());
        assert!(mock.execute_action(CultureAction::AdjustCo2).is_ok());
    }

    #[test]
    fn test_ipsc_protocol_steps() {
        let protocol = LabProtocol::ipsc_reprogramming();
        assert_eq!(protocol.steps.len(), 4);
        assert!(protocol.steps[0].requires_ethics_check);
        assert!(!protocol.steps[1].requires_ethics_check);
    }

    #[test]
    fn test_controller_single_step() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let step = ProtocolStep {
            name: "Test Step".to_string(),
            duration_seconds: 3600.0,
            min_viability: 0.5,
            expected_action: None,
            requires_ethics_check: false,
        };

        let result = controller.execute_step(&step, 0);
        assert!(result.action_executed);
        assert!(result.failure_reason.is_none());
        assert!(result.viability > 0.9);
    }

    #[test]
    fn test_controller_ethics_blocks_without_approval() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::new(); // No approvals
        let mut controller = LabController::new(mock, ethics);

        let step = ProtocolStep {
            name: "Factor Delivery".to_string(),
            duration_seconds: 3600.0,
            min_viability: 0.5,
            expected_action: None,
            requires_ethics_check: true,
        };

        let result = controller.execute_step(&step, 0);
        assert!(!result.action_executed);
        assert!(
            result
                .failure_reason
                .as_ref()
                .unwrap()
                .contains("Ethics gate")
        );
    }

    #[test]
    fn test_controller_viability_abort() {
        let mut cell = CellState::new_somatic();
        cell.viability = 0.1;
        let mock = MockInstrument::with_state(cell, CultureEnvironment::standard());
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let step = ProtocolStep {
            name: "Test".to_string(),
            duration_seconds: 3600.0,
            min_viability: 0.5,
            expected_action: None,
            requires_ethics_check: false,
        };

        let result = controller.execute_step(&step, 0);
        assert!(!result.action_executed);
        assert!(
            result
                .failure_reason
                .as_ref()
                .unwrap()
                .contains("Viability")
        );
    }

    #[test]
    fn test_run_protocol_success() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let protocol = LabProtocol {
            name: "Simple Test".to_string(),
            steps: vec![
                ProtocolStep {
                    name: "Step 1".to_string(),
                    duration_seconds: 3600.0,
                    min_viability: 0.5,
                    expected_action: None,
                    requires_ethics_check: false,
                },
                ProtocolStep {
                    name: "Step 2".to_string(),
                    duration_seconds: 3600.0,
                    min_viability: 0.5,
                    expected_action: None,
                    requires_ethics_check: false,
                },
            ],
            safety_viability_floor: 0.2,
            max_duration_seconds: 86_400.0,
        };

        let result = controller.run_protocol(&protocol);
        assert!(result.success);
        assert_eq!(result.steps_completed, 2);
    }

    #[test]
    fn test_run_protocol_fails_on_ethics() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::new();
        let mut controller = LabController::new(mock, ethics);

        let protocol = LabProtocol::ipsc_reprogramming();
        let result = controller.run_protocol(&protocol);
        assert!(!result.success);
        assert_eq!(result.steps_completed, 0);
    }

    #[test]
    fn test_instrument_name() {
        let mock = MockInstrument::new();
        assert_eq!(mock.name(), "MockInstrument");
    }

    #[test]
    fn test_predicted_state_similarity_bounded() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let step = ProtocolStep {
            name: "Test".to_string(),
            duration_seconds: 3600.0,
            min_viability: 0.5,
            expected_action: None,
            requires_ethics_check: false,
        };

        let result = controller.execute_step(&step, 0);
        assert!(result.predicted_state_similarity.is_finite());
    }

    // ── Track C: edge-path tests ─────────────────────────────────────────

    #[test]
    fn test_run_protocol_zero_steps() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let protocol = LabProtocol {
            name: "Empty".to_string(),
            steps: vec![],
            safety_viability_floor: 0.2,
            max_duration_seconds: 86_400.0,
        };

        let result = controller.run_protocol(&protocol);
        assert!(result.success);
        assert_eq!(result.steps_completed, 0);
        assert!(result.step_results.is_empty());
    }

    #[test]
    fn test_run_protocol_duration_limit() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let protocol = LabProtocol {
            name: "Long Protocol".to_string(),
            steps: vec![
                ProtocolStep {
                    name: "Step 1".to_string(),
                    duration_seconds: 100.0,
                    min_viability: 0.5,
                    expected_action: None,
                    requires_ethics_check: false,
                },
                ProtocolStep {
                    name: "Step 2".to_string(),
                    duration_seconds: 100.0,
                    min_viability: 0.5,
                    expected_action: None,
                    requires_ethics_check: false,
                },
                ProtocolStep {
                    name: "Step 3 — should be blocked by duration".to_string(),
                    duration_seconds: 100.0,
                    min_viability: 0.5,
                    expected_action: None,
                    requires_ethics_check: false,
                },
            ],
            safety_viability_floor: 0.2,
            max_duration_seconds: 150.0, // only enough for 1 step
        };

        let result = controller.run_protocol(&protocol);
        assert!(!result.success);
        assert!(
            result
                .failure_reason
                .as_ref()
                .unwrap()
                .contains("max duration")
        );
        // Step 0 completes; step 1 is rejected before execution because its
        // projected end time would exceed the protocol limit.
        assert_eq!(result.steps_completed, 1);
        assert_eq!(controller.instrument().executed_actions().len(), 1);
    }

    #[test]
    fn test_run_protocol_viability_floor() {
        // Viability is above per-step min but below protocol floor
        let mut cell = CellState::new_somatic();
        cell.viability = 0.15; // below protocol floor (0.5) but above step min (0.1)
        let mock = MockInstrument::with_state(cell, CultureEnvironment::standard());
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let protocol = LabProtocol {
            name: "Floor Test".to_string(),
            steps: vec![ProtocolStep {
                name: "Step 1".to_string(),
                duration_seconds: 3600.0,
                min_viability: 0.1, // step allows it
                expected_action: None,
                requires_ethics_check: false,
            }],
            safety_viability_floor: 0.5, // protocol floor catches it
            max_duration_seconds: 86_400.0,
        };

        let result = controller.run_protocol(&protocol);
        assert!(!result.success);
        assert!(
            result
                .failure_reason
                .as_ref()
                .unwrap()
                .contains("safety floor")
        );
    }

    #[test]
    fn test_execute_step_instrument_failure() {
        let mut env = CultureEnvironment::standard();
        env.temperature_celsius = 39.0;
        let mut mock = MockInstrument::with_state(CellState::new_somatic(), env);
        mock.set_fail_on(CultureAction::AdjustTemp);
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let step = ProtocolStep {
            name: "Test".to_string(),
            duration_seconds: 3600.0,
            min_viability: 0.5,
            expected_action: Some(CultureAction::AdjustTemp),
            requires_ethics_check: false,
        };

        let result = controller.execute_step(&step, 0);
        assert!(!result.action_executed);
        assert_eq!(result.selected_action, CultureAction::AdjustTemp);
        assert_eq!(result.action_matches_expectation, Some(true));
        assert!(result.failure_reason.is_some());
    }

    #[test]
    fn test_expected_action_mismatch_is_reported_without_overriding_controller() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);
        let step = ProtocolStep {
            name: "Expectation audit".to_string(),
            duration_seconds: 3600.0,
            min_viability: 0.5,
            expected_action: Some(CultureAction::AdjustTemp),
            requires_ethics_check: false,
        };

        let result = controller.execute_step(&step, 0);
        assert_eq!(result.selected_action, CultureAction::ExtendIncubation);
        assert_eq!(result.expected_action, Some(CultureAction::AdjustTemp));
        assert_eq!(result.action_matches_expectation, Some(false));
        assert!(result.action_executed);
    }

    #[test]
    fn test_invalid_step_duration_fails_closed() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);
        let step = ProtocolStep {
            name: "Invalid duration".to_string(),
            duration_seconds: f64::NAN,
            min_viability: 0.5,
            expected_action: None,
            requires_ethics_check: false,
        };

        let result = controller.execute_step(&step, 0);
        assert!(!result.action_executed);
        assert!(
            result
                .failure_reason
                .unwrap()
                .contains("Invalid step duration")
        );
    }

    #[test]
    fn test_protocol_exact_duration_limit_is_allowed() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);
        let protocol = LabProtocol {
            name: "Exact limit".to_string(),
            steps: vec![ProtocolStep {
                name: "Only step".to_string(),
                duration_seconds: 100.0,
                min_viability: 0.5,
                expected_action: None,
                requires_ethics_check: false,
            }],
            safety_viability_floor: 0.2,
            max_duration_seconds: 100.0,
        };

        let result = controller.run_protocol(&protocol);
        assert!(result.success);
        assert_eq!(result.steps_completed, 1);
    }

    #[test]
    fn test_instrument_mut_accessor() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        // Mutate instrument state through accessor
        controller.instrument_mut().set_cell_state({
            let mut cell = CellState::new_somatic();
            cell.viability = 0.3;
            cell
        });

        // Verify mutation took effect
        let cell = controller.instrument().read_cell_state();
        assert!((cell.viability - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_ipsc_protocol_field_values() {
        let protocol = LabProtocol::ipsc_reprogramming();
        assert_eq!(protocol.name, "iPSC Reprogramming");
        assert_eq!(protocol.steps.len(), 4);
        assert!((protocol.safety_viability_floor - 0.2).abs() < 1e-6);
        assert!((protocol.max_duration_seconds - 2_419_200.0).abs() < 1e-6);

        // Step 0: Factor Delivery — requires ethics, 1 day
        assert!(protocol.steps[0].requires_ethics_check);
        assert!((protocol.steps[0].duration_seconds - 86_400.0).abs() < 1e-6);
        assert!((protocol.steps[0].min_viability - 0.8).abs() < 1e-6);

        // Step 3: Colony Stabilization — expects PassageCells
        assert_eq!(
            protocol.steps[3].expected_action,
            Some(CultureAction::PassageCells)
        );
    }

    #[test]
    fn test_run_full_ipsc_protocol_with_approval() {
        let mock = MockInstrument::new();
        let ethics = EthicsGate::fully_approved();
        let mut controller = LabController::new(mock, ethics);

        let protocol = LabProtocol::ipsc_reprogramming();
        let result = controller.run_protocol(&protocol);
        assert!(result.success);
        assert_eq!(result.steps_completed, 4);
        assert_eq!(result.step_results.len(), 4);

        // Verify step results have expected metadata
        for (i, sr) in result.step_results.iter().enumerate() {
            assert_eq!(sr.step_index, i);
            assert!(sr.action_executed);
            assert!(sr.failure_reason.is_none());
            assert!(sr.viability > 0.0);
            assert!(sr.predicted_state_similarity.is_finite());
        }
    }
}
