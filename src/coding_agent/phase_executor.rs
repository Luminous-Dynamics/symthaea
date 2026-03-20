use super::types::*;
use super::CodingAgent;
use crate::action::primitives::{Atom, Molecule, MoleculeExecutor, PrimitiveValue};
use crate::action::ActionOutcome;
use crate::cognitive_loop::motor_output_bridge::{ActionType, MotorOutputResult};
use crate::cognitive_loop::CycleResult;
use crate::consciousness::fep_active_inference::MotorCommandType;
use crate::language::intelligent_dispatcher::BackendTier;
use crate::mind::structured_thought::EpistemicStatus;
use std::path::PathBuf;

impl CodingAgent {
    /// Process the results of a cognitive cycle and decide the next phase.
    pub(crate) fn process_step_result(
        &mut self,
        cycle_result: &CycleResult,
        motor_result: Option<MotorOutputResult>,
        phi: f32,
    ) {
        let confidence = self.cognitive_loop.prediction_confidence();
        let epistemic = Self::confidence_to_epistemic(confidence);

        if self.phase == TaskPhase::Generating
            && epistemic == EpistemicStatus::Unknown
            && !self.generation_tiers.is_empty()
        {
            self.observations
                .push("Epistemic gate: confidence too low for generation, re-planning".into());
            self.phase = TaskPhase::Planning;
            self.phase_failures += 1;
            return;
        }

        if let Some(ref result) = motor_result {
            self.process_motor_result(result);
        }

        let fep_command = MotorCommandType::from_action_index(cycle_result.metadata.fep.fep_action);

        let suppress_exploration = self.iteration >= 3 && self.generation_tiers.is_empty();
        let has_code = self.generated_code.is_some();
        let in_action_phase = matches!(
            self.phase,
            TaskPhase::Generating | TaskPhase::Testing | TaskPhase::Fixing
        );

        if self.phase != TaskPhase::Done {
            match fep_command {
                MotorCommandType::ExplorationTrigger => {
                    if self.phase != TaskPhase::Understanding
                        && !suppress_exploration
                        && !(has_code && in_action_phase)
                    {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            from = %self.phase,
                            "FEP ExplorationTrigger → Understanding"
                        );
                        self.phase = TaskPhase::Understanding;
                        self.phase_failures = 0;
                        return;
                    } else if suppress_exploration {
                        tracing::debug!(
                            target: "symthaea::coding_agent",
                            iteration = self.iteration,
                            "Suppressing FEP ExplorationTrigger — need to attempt generation"
                        );
                    }
                }
                MotorCommandType::ReflectionInitiate => {
                    if self.phase != TaskPhase::Planning
                        && self.phase != TaskPhase::Understanding
                        && !(has_code && in_action_phase)
                    {
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            from = %self.phase,
                            "FEP ReflectionInitiate → Planning"
                        );
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                        return;
                    }
                }
                MotorCommandType::ExpectationReset => {
                    if (self.phase == TaskPhase::Generating || self.phase == TaskPhase::Fixing)
                        && !has_code
                    {
                        self.observations
                            .push("FEP ExpectationReset: model mismatch, re-planning".into());
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                        return;
                    }
                }
                MotorCommandType::MemoryConsolidate => {
                    self.observations
                        .push("FEP MemoryConsolidate: consolidating learned patterns".into());
                }
                _ => {}
            }
        }

        // ── Code Quality Gate ─────────────────────────────────────────
        let is_rust = self.target_language() == "rust";
        if is_rust
            && (self.phase == TaskPhase::Generating || self.phase == TaskPhase::Fixing)
            && self.generated_code.is_some()
        {
            if let Some(ref code) = self.generated_code {
                if let Some(quality_issue) = Self::check_code_quality(code) {
                    tracing::info!(
                        target: "symthaea::coding_agent",
                        issue = %quality_issue,
                        "Code quality gate: rejecting generated code"
                    );
                    self.quality_rejections += 1;
                    let rejection_pattern = format!("quality_gate: {}", quality_issue);
                    if let Some((_, count)) = self
                        .failure_patterns
                        .iter_mut()
                        .find(|(p, _)| *p == rejection_pattern)
                    {
                        *count += 1;
                    } else {
                        self.failure_patterns.push((rejection_pattern, 1));
                    }
                    self.observations
                        .push(format!("Quality gate rejected code: {quality_issue}"));
                    if let Some(tier) = self.generation_tiers.last().copied() {
                        if let Some(ref mut dispatcher) = self.dispatcher {
                            dispatcher.record_outcome(tier, false);
                        }
                    }
                    self.generated_code = None;
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                    }
                    return;
                }
            }
        }

        // Force-advance
        if self.iteration >= 4
            && self.generation_tiers.is_empty()
            && self.phase != TaskPhase::Done
            && self.phase != TaskPhase::Generating
        {
            tracing::info!(
                target: "symthaea::coding_agent",
                iteration = self.iteration,
                phase = %self.phase,
                "Force-advancing to Generating"
            );
            self.phase = TaskPhase::Generating;
            self.phase_failures = 0;
            self.generated_code = None;
            return;
        }

        // Default phase transitions
        match self.phase {
            TaskPhase::Understanding => {
                if self.iteration >= 1 || !self.observations.is_empty() {
                    self.phase = TaskPhase::Planning;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Planning");
                }
            }
            TaskPhase::Planning => {
                self.phase = TaskPhase::Generating;
                self.phase_failures = 0;
                self.generated_code = None;
                tracing::info!(target: "symthaea::coding_agent", "→ Generating");
            }
            TaskPhase::Generating => {
                let code_written = self.generated_code.is_some();
                let check_passed = motor_result.as_ref().map_or(false, |r| r.success);

                if code_written && check_passed {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Testing");
                } else if code_written {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "→ Testing (unverified)");
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Planning;
                        self.phase_failures = 0;
                        tracing::warn!(
                            target: "symthaea::coding_agent",
                            "Generation failed {} times, re-planning",
                            self.config.max_phase_failures
                        );
                    }
                }
            }
            TaskPhase::Testing => {
                let effective_result = motor_result.clone().or_else(|| {
                    if self.generated_code.is_some() {
                        self.do_testing_molecule()
                    } else {
                        None
                    }
                });

                if let Some(ref result) = effective_result {
                    self.record_generation_outcome(result.success);

                    if result.success {
                        self.tests_passed = Some(true);
                        self.phase = TaskPhase::Done;
                        tracing::info!(target: "symthaea::coding_agent", "→ Done (tests passed)");
                    } else {
                        self.tests_passed = Some(false);

                        let stuck_on_error = if let Some(ref output) = self.last_test_output {
                            let norm = Self::normalize_error_pattern(output);
                            self.failure_patterns
                                .iter()
                                .find(|(p, _)| *p == norm)
                                .map(|(_, c)| *c >= 3)
                                .unwrap_or(false)
                        } else {
                            false
                        };

                        if stuck_on_error {
                            self.stuck_detected = true;
                            let strategy = self.next_retry_strategy();
                            self.retry_state.current_strategy = strategy;
                            tracing::info!(
                                target: "symthaea::coding_agent",
                                strategy = ?self.retry_state.current_strategy,
                                "Stuck detection: same error 3+ times, escalating"
                            );
                        }

                        self.phase = TaskPhase::Fixing;
                        self.phase_failures = 0;
                        self.generated_code = None;
                        tracing::info!(target: "symthaea::coding_agent", "→ Fixing");
                    }
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        self.phase = TaskPhase::Done;
                    }
                }
            }
            TaskPhase::Fixing => {
                let code_written = self.generated_code.is_some();
                let effective_result = motor_result.clone().or_else(|| {
                    if code_written {
                        self.do_testing_molecule()
                    } else {
                        None
                    }
                });
                if let Some(ref result) = effective_result {
                    if result.success || code_written {
                        self.phase = TaskPhase::Testing;
                        self.phase_failures = 0;
                        tracing::info!(target: "symthaea::coding_agent", "→ Testing (after fix)");
                    } else {
                        self.phase_failures += 1;
                        if self.phase_failures >= self.config.max_phase_failures {
                            let strategy = self.next_retry_strategy();
                            match strategy {
                                RetryStrategy::RequestClarification(ref msg) => {
                                    self.emit_event(AgentEvent::RequestClarification(msg.clone()));
                                    self.phase = TaskPhase::Done;
                                    tracing::warn!(
                                        target: "symthaea::coding_agent",
                                        "All retry strategies exhausted, requesting clarification"
                                    );
                                }
                                _ => {
                                    self.retry_state.current_strategy = strategy;
                                    self.phase = TaskPhase::Planning;
                                    self.phase_failures = 0;
                                    self.generated_code = None;
                                    tracing::info!(
                                        target: "symthaea::coding_agent",
                                        strategy = ?self.retry_state.current_strategy,
                                        "Retry strategy: re-planning with different approach"
                                    );
                                }
                            }
                        }
                    }
                } else if code_written {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                } else {
                    self.phase_failures += 1;
                    if self.phase_failures >= self.config.max_phase_failures {
                        let strategy = self.next_retry_strategy();
                        match strategy {
                            RetryStrategy::RequestClarification(_) => {
                                self.phase = TaskPhase::Done;
                            }
                            _ => {
                                self.retry_state.current_strategy = strategy;
                                self.phase = TaskPhase::Planning;
                                self.phase_failures = 0;
                                self.generated_code = None;
                            }
                        }
                    }
                }
            }
            TaskPhase::Done => {} // terminal
        }

        // Stuck detection: low Phi
        if self.phi_trace.len() >= 3
            && self.phase != TaskPhase::Done
            && self.phase != TaskPhase::Understanding
            && self.phase != TaskPhase::Planning
            && self.phase != TaskPhase::Testing
            && self.phase != TaskPhase::Fixing
            && !self.generation_tiers.is_empty()
        {
            let recent: Vec<f32> = self.phi_trace.iter().rev().take(3).copied().collect();
            let all_low = recent.iter().all(|&p| p < 0.2);
            if all_low {
                self.observations.push(
                    "Stuck detection: Phi consistently low, trying different approach".into(),
                );
                self.phase = TaskPhase::Planning;
                self.phase_failures = 0;
            }
        }
    }

    /// Process a motor output result — extract observations, track files.
    pub(crate) fn process_motor_result(&mut self, result: &MotorOutputResult) {
        if result.success {
            if let Some(ref outcome) = result.outcome {
                match outcome {
                    ActionOutcome::FileContent(data) => {
                        let content =
                            String::from_utf8_lossy(&data[..data.len().min(2000)]).to_string();
                        self.observations.push(format!(
                            "Read file ({} bytes): {}",
                            data.len(),
                            &content[..content.len().min(200)]
                        ));
                    }
                    ActionOutcome::DirectoryListing(entries) => {
                        let listing: Vec<String> = entries
                            .iter()
                            .take(20)
                            .map(|p| p.display().to_string())
                            .collect();
                        self.observations
                            .push(format!("Directory listing: {:?}", listing));
                    }
                    ActionOutcome::Success => {
                        if let Some(ActionType::Write) = result.action_type {
                            self.observations.push("File written successfully".into());
                        } else if let Some(ActionType::CargoCheck) | Some(ActionType::CargoTest) =
                            result.action_type
                        {
                            self.observations.push("Check/test passed".into());
                        } else {
                            self.observations.push("Action succeeded".into());
                        }
                    }
                    _ => {
                        self.observations
                            .push(format!("Action result: {:?}", result.action_type));
                    }
                }
            }
        } else if let Some(ref error) = result.error {
            self.errors.push(error.clone());

            if result.action_type == Some(ActionType::CargoTest)
                || result.action_type == Some(ActionType::CargoCheck)
            {
                self.last_test_output = Some(error.clone());

                let pattern = Self::normalize_error_pattern(error);
                if let Some(entry) = self
                    .failure_patterns
                    .iter_mut()
                    .find(|(p, _)| *p == pattern)
                {
                    entry.1 += 1;
                } else {
                    self.failure_patterns.push((pattern.clone(), 1));
                }

                self.store_experience(error, false);
            }
        }
    }

    /// Execute a molecule through MoleculeExecutor and convert results.
    pub(crate) fn execute_molecule(&mut self, molecule: &Molecule) -> Option<MotorOutputResult> {
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        let real_exec = self.config.enable_real_exec;
        let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, real_exec);

        match executor.execute(molecule) {
            Ok(val) => {
                self.energy_budget = executor.energy_budget;
                self.store_execution_trace(&executor.trace);

                match &val {
                    PrimitiveValue::Text(text) => {
                        if !text.is_empty() && text.len() <= 2000 {
                            self.observations.push(text.clone());
                        } else if text.len() > 2000 {
                            self.observations.push(format!(
                                "{}...(truncated, {} bytes total)",
                                &text[..1500],
                                text.len()
                            ));
                        }
                    }
                    PrimitiveValue::Listing(paths) => {
                        let names: Vec<String> = paths
                            .iter()
                            .take(50)
                            .map(|p| {
                                p.file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| p.display().to_string())
                            })
                            .collect();
                        self.observations
                            .push(format!("Files: [{}]", names.join(", ")));
                    }
                    PrimitiveValue::CommandResult {
                        stdout,
                        stderr,
                        exit_code,
                    } => {
                        let success = *exit_code == 0;
                        if !success {
                            self.last_test_output = Some(stderr.clone());
                            self.observations.push(format!(
                                "Command failed (exit={}):\n{}",
                                exit_code,
                                &stderr[..stderr.len().min(500)]
                            ));
                        } else {
                            self.observations
                                .push(format!("Command succeeded (exit={})", exit_code));
                        }
                        return Some(MotorOutputResult {
                            success,
                            action_type: Some(ActionType::CargoCheck),
                            prediction_error: if success { 0.0 } else { 0.8 },
                            outcome: Some(ActionOutcome::CommandOutput {
                                stdout: stdout.as_bytes().to_vec(),
                                stderr: stderr.as_bytes().to_vec(),
                                exit_code: *exit_code,
                            }),
                            error: if success { None } else { Some(stderr.clone()) },
                        });
                    }
                    _ => {}
                }

                Some(MotorOutputResult {
                    success: true,
                    action_type: None,
                    prediction_error: 0.0,
                    outcome: Some(ActionOutcome::Success),
                    error: None,
                })
            }
            Err(e) => {
                let error_msg = format!("{}", e);
                tracing::warn!(
                    target: "symthaea::coding_agent",
                    error = %error_msg,
                    "Molecule execution failed"
                );
                self.observations
                    .push(format!("Execution error: {}", error_msg));
                Some(MotorOutputResult {
                    success: false,
                    action_type: None,
                    prediction_error: 1.0,
                    outcome: None,
                    error: Some(error_msg),
                })
            }
        }
    }

    /// Execute the Understanding phase via molecules.
    pub(crate) fn do_understanding_molecule(&mut self) {
        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);

        // 1. List working directory
        {
            let mol = Molecule::atom(Atom::list(working_dir.clone()));
            let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, true);
            if let Ok(PrimitiveValue::Listing(paths)) = executor.execute(&mol) {
                self.energy_budget = executor.energy_budget;
                let mut names: Vec<String> = paths
                    .iter()
                    .filter_map(|p| {
                        let name = p.file_name()?.to_string_lossy().to_string();
                        if name.starts_with('.') || name == "target" || name == "node_modules" {
                            None
                        } else if p.is_dir() {
                            Some(format!("{}/", name))
                        } else {
                            Some(name)
                        }
                    })
                    .collect();
                names.sort();
                if !names.is_empty() {
                    self.observations.push(format!(
                        "Working directory {}: [{}]",
                        working_dir.display(),
                        names.join(", ")
                    ));
                }
            }
        }

        // 2. Read target file
        if target.exists() {
            let mol = Molecule::atom(Atom::read(target.clone()));
            let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, true);
            if let Ok(PrimitiveValue::Text(content)) = executor.execute(&mol) {
                self.energy_budget = executor.energy_budget;
                let preview = if content.len() > 1500 {
                    format!("{}...(truncated)", &content[..1500])
                } else {
                    content
                };
                self.observations.push(format!(
                    "Target file {} ({} bytes):\n{}",
                    target.display(),
                    preview.len(),
                    preview
                ));
            }
        } else {
            self.observations.push(format!(
                "Target file {} does not exist yet (will be created)",
                target.display()
            ));
        }

        // 3. Read Cargo.toml
        let cargo_toml = working_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let mol = Molecule::atom(Atom::read(cargo_toml));
            let mut executor = MoleculeExecutor::new(current_phi, self.energy_budget, true);
            if let Ok(PrimitiveValue::Text(content)) = executor.execute(&mol) {
                self.energy_budget = executor.energy_budget;
                let preview: String = content.lines().take(15).collect::<Vec<_>>().join("\n");
                self.observations.push(format!("Cargo.toml:\n{}", preview));
            }
        }

        // Query experience store
        let hints = self.retrieve_experience_hints();
        if !hints.is_empty() {
            self.observations.push(format!(
                "Prior experience: {} relevant patterns",
                hints.len()
            ));
            for (pattern, hint) in hints.iter().take(3) {
                self.observations.push(format!(
                    "  Prior: {} → {}",
                    &pattern[..pattern.len().min(80)],
                    &hint[..hint.len().min(80)]
                ));
            }
        }
    }

    /// Execute the Testing phase via molecules.
    pub(crate) fn do_testing_molecule(&mut self) -> Option<MotorOutputResult> {
        if !self.config.enable_real_exec {
            return Some(MotorOutputResult {
                success: self.generated_code.is_some(),
                action_type: Some(ActionType::CargoCheck),
                prediction_error: 0.0,
                outcome: Some(ActionOutcome::Success),
                error: None,
            });
        }

        let working_dir = self.config.working_dir.clone();
        if !working_dir.join("Cargo.toml").exists() {
            return Some(MotorOutputResult {
                success: false,
                action_type: Some(ActionType::CargoCheck),
                prediction_error: 0.5,
                outcome: None,
                error: Some("No Cargo.toml in working directory".into()),
            });
        }

        let molecule = self
            .current_plan
            .as_ref()
            .and_then(|profile| {
                if profile.atom_names.contains(&"CargoTest") {
                    Some(Molecule::atom(Atom::cargo_test(working_dir.clone())))
                } else if profile.atom_names.contains(&"CargoCheck") {
                    Some(Molecule::atom(Atom::cargo_check(working_dir.clone())))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| Molecule::atom(Atom::cargo_check(working_dir)));

        self.execute_molecule(&molecule)
    }
}
