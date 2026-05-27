// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FEP planning and execution for the coding agent.

use super::*;

impl CodingAgent {
    /// Build the observation text that the cognitive loop will process.
    pub(super) fn build_observation(&self) -> String {
        let mut obs = format!("CODING TASK: {}\n", self.task);
        obs.push_str(&format!("PHASE: {}\n", self.phase));
        obs.push_str(&format!(
            "ITERATION: {}/{}\n",
            self.iteration, self.config.max_iterations
        ));

        if !self.observations.is_empty() {
            obs.push_str("CONTEXT:\n");
            for o in self.observations.iter().rev().take(3).rev() {
                obs.push_str(&format!("  {}\n", o));
            }
        }

        if let Some(ref test_output) = self.last_test_output {
            obs.push_str(&format!("LAST TEST OUTPUT:\n{}\n", test_output));
        }

        if !self.errors.is_empty() {
            obs.push_str("ERRORS:\n");
            for e in self.errors.iter().rev().take(2) {
                obs.push_str(&format!("  {}\n", e));
            }
        }

        obs
    }

    /// Build a typed execution plan for the current phase.
    pub(super) fn build_execution_plan(&self) -> Option<Molecule> {
        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();

        match self.phase {
            TaskPhase::Understanding => {
                let mut plan = Molecule::atom(Atom::list(working_dir.clone()));
                if target.exists() {
                    plan = plan.then(Molecule::atom(Atom::read(target)));
                }
                Some(plan)
            }
            TaskPhase::Planning => None,
            TaskPhase::Generating => {
                if let Some(ref code) = self.generated_code {
                    Some(crate::action::primitives::recipes::write_and_check(
                        target, code,
                    ))
                } else {
                    None
                }
            }
            TaskPhase::Testing => Some(Molecule::atom(Atom::cargo_check(working_dir))),
            TaskPhase::Fixing => {
                if let Some(ref code) = self.generated_code {
                    let write_check =
                        crate::action::primitives::recipes::write_and_check(target, code);
                    Some(write_check.recover(|_| Molecule::atom(Atom::Noop)))
                } else {
                    None
                }
            }
            TaskPhase::Done => None,
        }
    }

    /// Generate multiple candidate plans and use FEP free-energy minimization to select the best.
    pub(super) fn select_plan_fep(&self) -> Option<(Molecule, PlanProfile)> {
        use crate::action::primitives::{
            PlanCandidate, select_best_plan, select_best_plan_with_history,
        };

        let target = self.resolve_target_file();
        let working_dir = self.config.working_dir.clone();
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);

        let candidates: Vec<PlanCandidate> = match self.phase {
            TaskPhase::Understanding => {
                let mut plans = vec![];
                let list_only = Molecule::atom(Atom::list(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "list_only".into(),
                    profile: list_only.profile(),
                    molecule: list_only,
                });
                if target.exists() {
                    let list_and_read = Molecule::atom(Atom::list(working_dir.clone()))
                        .then(Molecule::atom(Atom::read(target.clone())));
                    plans.push(PlanCandidate {
                        name: "list_and_read".into(),
                        profile: list_and_read.profile(),
                        molecule: list_and_read,
                    });
                }
                let cargo_toml = working_dir.join("Cargo.toml");
                if cargo_toml.exists() && target.exists() {
                    let gather = Molecule::atom(Atom::read(cargo_toml))
                        .then(Molecule::atom(Atom::read(target.clone())))
                        .then(Molecule::atom(Atom::list(working_dir.clone())));
                    plans.push(PlanCandidate {
                        name: "full_context".into(),
                        profile: gather.profile(),
                        molecule: gather,
                    });
                }
                plans
            }
            TaskPhase::Testing => {
                let mut plans = vec![];
                let check = Molecule::atom(Atom::cargo_check(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "cargo_check".into(),
                    profile: check.profile(),
                    molecule: check,
                });
                let check_clippy = Molecule::atom(Atom::cargo_check(working_dir.clone()))
                    .then(Molecule::atom(Atom::cargo_clippy(working_dir.clone())));
                plans.push(PlanCandidate {
                    name: "check_and_clippy".into(),
                    profile: check_clippy.profile(),
                    molecule: check_clippy,
                });
                let test = Molecule::atom(Atom::cargo_test(working_dir.clone()));
                plans.push(PlanCandidate {
                    name: "cargo_test".into(),
                    profile: test.profile(),
                    molecule: test,
                });
                plans
            }
            TaskPhase::Generating => {
                let mut plans = vec![];
                if let Some(ref code) = self.generated_code {
                    let wc =
                        crate::action::primitives::recipes::write_and_check(target.clone(), code);
                    plans.push(PlanCandidate {
                        name: "write_and_check".into(),
                        profile: wc.profile(),
                        molecule: wc,
                    });
                    let wct = crate::action::primitives::recipes::full_coding_workflow(
                        target.clone(),
                        code.clone(),
                        working_dir.clone(),
                    );
                    plans.push(PlanCandidate {
                        name: "full_workflow".into(),
                        profile: wct.profile(),
                        molecule: wct,
                    });
                }
                let prompt = self.build_generation_prompt();
                for (name, mol) in crate::action::primitives::recipes::tiered_generation_candidates(
                    target.clone(),
                    &prompt,
                ) {
                    plans.push(PlanCandidate {
                        name,
                        profile: mol.profile(),
                        molecule: mol,
                    });
                }
                plans
            }
            TaskPhase::Fixing => {
                if let Some(ref code) = self.generated_code {
                    let mut plans = vec![];
                    let wc =
                        crate::action::primitives::recipes::write_and_check(target.clone(), code);
                    plans.push(PlanCandidate {
                        name: "fix_and_check".into(),
                        profile: wc.profile(),
                        molecule: wc,
                    });
                    let wcr =
                        crate::action::primitives::recipes::write_and_check(target.clone(), code)
                            .recover(|_| Molecule::atom(Atom::Noop));
                    plans.push(PlanCandidate {
                        name: "fix_with_recovery".into(),
                        profile: wcr.profile(),
                        molecule: wcr,
                    });
                    plans
                } else {
                    vec![]
                }
            }
            TaskPhase::Planning | TaskPhase::Done => vec![],
        };

        if candidates.is_empty() {
            return None;
        }

        let selected_idx = if let Some(ref store) = self.experience_store {
            let recipe_keys: Vec<&str> = candidates
                .iter()
                .map(|c| c.profile.atom_names.first().copied().unwrap_or("Unknown"))
                .collect();
            let rates = store.recipe_success_rates(&recipe_keys);
            select_best_plan_with_history(&candidates, current_phi, self.energy_budget, &rates)
        } else {
            select_best_plan(&candidates, current_phi, self.energy_budget)
        };

        let selected_idx = selected_idx?;
        let selected = &candidates[selected_idx];

        tracing::debug!(
            target: "symthaea::coding_agent",
            phase = %self.phase,
            selected = %selected.name,
            candidates = candidates.len(),
            energy = selected.profile.total_energy,
            "FEP selected plan (history-aware)"
        );

        let profile = selected.profile.clone();
        self.build_execution_plan().map(|m| (m, profile))
    }

    /// Evaluate whether the current plan is safe and affordable.
    pub(super) fn evaluate_plan(&self, plan: &Molecule, current_phi: f32) -> (bool, String) {
        let profile = plan.profile();

        if !profile.phi_sufficient(current_phi) {
            return (
                false,
                format!(
                    "Phi too low: {:.3} < {:.3} required",
                    current_phi, profile.min_phi
                ),
            );
        }

        if !profile.within_budget(self.energy_budget) {
            return (
                false,
                format!(
                    "Energy budget exceeded: plan costs {:.1}, budget remaining {:.1}",
                    profile.total_energy, self.energy_budget
                ),
            );
        }

        if profile.max_destructiveness == crate::action::DestructivenessLevel::Destructive {
            return (
                false,
                format!(
                    "Plan contains destructive action ({}) — requires confirmation",
                    profile.atom_names.join(" -> ")
                ),
            );
        }

        (
            true,
            format!(
                "Plan approved: {} steps, energy {:.1}/{:.1}, phi {:.3}/{:.3}",
                profile.step_count,
                profile.total_energy,
                self.energy_budget,
                current_phi,
                profile.min_phi,
            ),
        )
    }

    /// Deduct energy cost from the budget after execution.
    pub(super) fn deduct_energy(&mut self, profile: &PlanProfile) {
        self.energy_budget = (self.energy_budget - profile.total_energy).max(0.0);
    }

    pub(super) fn build_motor_request(&self) -> MotorActionRequest {
        match self.phase {
            TaskPhase::Understanding => MotorActionRequest {
                target_path: Some(self.config.working_dir.clone()),
                ..Default::default()
            },
            TaskPhase::Planning => MotorActionRequest::default(),
            TaskPhase::Generating => {
                if self.generated_code.is_some() {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        program: Some("cargo".into()),
                        args: vec!["check".into()],
                        ..Default::default()
                    }
                } else {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        ..Default::default()
                    }
                }
            }
            TaskPhase::Testing => MotorActionRequest {
                target_path: Some(self.config.working_dir.clone()),
                program: Some("cargo".into()),
                args: vec!["check".into()],
                ..Default::default()
            },
            TaskPhase::Fixing => {
                if self.generated_code.is_some() {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        program: Some("cargo".into()),
                        args: vec!["check".into()],
                        ..Default::default()
                    }
                } else {
                    MotorActionRequest {
                        target_path: Some(self.config.working_dir.clone()),
                        ..Default::default()
                    }
                }
            }
            TaskPhase::Done => MotorActionRequest::default(),
        }
    }

    /// Process the results of a cognitive cycle and decide the next phase.
    pub(super) fn process_step_result(
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
                            "FEP ExplorationTrigger -> Understanding"
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
                            "FEP ReflectionInitiate -> Planning"
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

        // Code Quality Gate
        if (self.phase == TaskPhase::Generating || self.phase == TaskPhase::Fixing)
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
                            dispatcher.record_outcome_with_category(tier, false, &self.task);
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
                    tracing::info!(target: "symthaea::coding_agent", "-> Planning");
                }
            }
            TaskPhase::Planning => {
                self.phase = TaskPhase::Generating;
                self.phase_failures = 0;
                self.generated_code = None;
                tracing::info!(target: "symthaea::coding_agent", "-> Generating");
            }
            TaskPhase::Generating => {
                let code_written = self.generated_code.is_some();
                let check_passed = motor_result.as_ref().map_or(false, |r| r.success);

                if code_written && check_passed {
                    // Run GCS verification on compiled code before advancing
                    if let Some(ref code) = self.generated_code {
                        let violations = self.verify_with_gcs(code, &self.task);
                        if !violations.is_empty() {
                            tracing::warn!(
                                target: "symthaea::coding_agent::gcs",
                                count = violations.len(),
                                "GCS verification found violations — routing to Fixing"
                            );
                            for v in &violations {
                                self.observations.push(format!("GCS: {}", v));
                            }
                            self.gcs_violations = violations;
                            self.phase = TaskPhase::Fixing;
                            self.phase_failures = 0;
                        } else {
                            self.gcs_violations.clear();
                            self.phase = TaskPhase::Testing;
                            self.phase_failures = 0;
                            tracing::info!(target: "symthaea::coding_agent", "-> Testing");
                        }
                    } else {
                        self.phase = TaskPhase::Testing;
                        self.phase_failures = 0;
                        tracing::info!(target: "symthaea::coding_agent", "-> Testing");
                    }
                } else if code_written {
                    self.phase = TaskPhase::Testing;
                    self.phase_failures = 0;
                    tracing::info!(target: "symthaea::coding_agent", "-> Testing (unverified)");
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

                        // Backfill error knowledge: now that tests passed, update all
                        // recorded fix facts with tests_passed=true. This closes the
                        // learning loop — Bayesian success rates improve over time.
                        self.backfill_error_knowledge_test_success();

                        tracing::info!(target: "symthaea::coding_agent", "-> Done (tests passed)");
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
                        tracing::info!(target: "symthaea::coding_agent", "-> Fixing");
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
                        tracing::info!(target: "symthaea::coding_agent", "-> Testing (after fix)");
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
            TaskPhase::Done => {}
        }

        // Stuck detection via low Phi
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
    pub(super) fn process_motor_result(&mut self, result: &MotorOutputResult) {
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

    /// Execute a molecule through MoleculeExecutor.
    pub(super) fn execute_molecule(&mut self, molecule: &Molecule) -> Option<MotorOutputResult> {
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
    pub(super) fn do_understanding_molecule(&mut self) {
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
                    "  Prior: {} -> {}",
                    &pattern[..pattern.len().min(80)],
                    &hint[..hint.len().min(80)]
                ));
            }
        }
    }

    /// Execute the Testing phase via molecules.
    pub(super) fn do_testing_molecule(&mut self) -> Option<MotorOutputResult> {
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
