// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Experience store and telemetry for the coding agent.

use super::*;

impl CodingAgent {
    /// Store a coding experience (success or failure) in the persistent store.
    pub(super) fn store_experience(&mut self, detail: &str, success: bool) {
        let experience = CodingExperience {
            task: self.task.clone(),
            detail: detail.chars().take(500).collect(),
            success,
            tier: self
                .generation_tiers
                .last()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            fix_hint: None,
        };

        if let Some(ref mut store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(async {
                    store.store(experience).await;
                });
            }
        }
    }

    /// Record the outcome of code generation into the dispatcher's Bayesian stats.
    pub(super) fn record_generation_outcome(&mut self, success: bool) {
        let tier = self.generation_tiers.last().copied();
        if let (Some(tier), Some(ref mut dispatcher)) = (tier, &mut self.dispatcher) {
            dispatcher.record_outcome_with_category(tier, success, &self.task);
        }

        if success {
            if let Some(code) = self.generated_code.clone() {
                let summary: String = code.chars().take(200).collect();
                self.store_experience(&summary, true);

                if let Some((last_error, _)) = self.failure_patterns.last().cloned() {
                    self.store_fix_hint(&last_error, &code);
                }

                if let Some(ref mut store) = self.experience_store {
                    store.store_learned_template(&self.task, &code, None);
                    tracing::info!(
                        target: "symthaea::coding_agent",
                        task = %self.task,
                        code_len = code.len(),
                        tier = ?tier,
                        "Stored learned template in experience store"
                    );
                }
            }
        }
    }

    /// Store a fix hint: when a failure pattern was resolved by a successful generation.
    pub(super) fn store_fix_hint(&mut self, error_pattern: &str, fix_code: &str) {
        let fix_summary: String = fix_code.lines().take(5).collect::<Vec<_>>().join("\n");
        let experience = CodingExperience {
            task: self.task.clone(),
            detail: error_pattern.chars().take(300).collect(),
            success: true,
            tier: self
                .generation_tiers
                .last()
                .map(|t| t.to_string())
                .unwrap_or_default(),
            fix_hint: Some(fix_summary),
        };

        if let Some(ref mut store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(async {
                    store.store(experience).await;
                });
            }
        }
    }

    /// Store a molecule execution trace.
    pub(super) fn store_execution_trace(&mut self, trace: &[(String, f32, String)]) {
        if trace.is_empty() {
            return;
        }

        let trace_summary: String = trace
            .iter()
            .map(|(name, energy, summary)| format!("  {} (E={:.1}): {}", name, energy, summary))
            .collect::<Vec<_>>()
            .join("\n");

        tracing::debug!(
            target: "symthaea::coding_agent",
            steps = trace.len(),
            "Molecule trace:\n{}", trace_summary
        );

        let total_energy: f32 = trace.iter().map(|(_, e, _)| e).sum();
        let atom_names: Vec<&str> = trace.iter().map(|(n, _, _)| n.as_str()).collect();
        let recipe_key = atom_names.join("\u{2192}");

        let last_success = trace
            .last()
            .map(|(_, _, s)| s.contains("exit=0") || s == "()")
            .unwrap_or(false);

        let experience = CodingExperience {
            task: format!("recipe:{}", recipe_key),
            detail: format!(
                "energy={:.1}, steps={}, atoms=[{}]",
                total_energy,
                trace.len(),
                recipe_key,
            ),
            success: last_success,
            tier: "MoleculeExecutor".to_string(),
            fix_hint: None,
        };

        if let Some(ref mut store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(async {
                    store.store(experience).await;
                });
            }
        }
    }

    /// Map prediction confidence to epistemic status.
    pub(super) fn confidence_to_epistemic(confidence: f32) -> EpistemicStatus {
        if confidence > 0.9 {
            EpistemicStatus::Certain
        } else if confidence > 0.7 {
            EpistemicStatus::Probable
        } else if confidence > 0.4 {
            EpistemicStatus::Uncertain
        } else {
            EpistemicStatus::Unknown
        }
    }

    /// Build the final result.
    pub(super) fn build_result(&self) -> AgentResult {
        let confidence = self.cognitive_loop.prediction_confidence();
        AgentResult {
            files_modified: self.files_modified.clone(),
            tests_passed: self.tests_passed,
            iterations_used: self.iteration,
            phi_trace: self.phi_trace.clone(),
            epistemic_status: Self::confidence_to_epistemic(confidence),
            final_phase: self.phase.clone(),
            observations: self.observations.clone(),
            errors: self.errors.clone(),
            generation_tiers: self.generation_tiers.clone(),
            total_energy: self.dispatcher.as_ref().map_or(0.0, |d| d.total_energy()),
            remaining_energy: self.energy_budget,
            failure_pattern_count: self.failure_patterns.len(),
            dedup_skips: self.dedup_skips,
            quality_rejections: self.quality_rejections,
            consciousness_deferrals: self.consciousness_deferrals,
            stuck_detected: self.stuck_detected,
            #[cfg(feature = "school_learning")]
            generated_lessons: self.generate_lessons_from_failures(),
        }
    }

    /// Generate auto-curriculum lessons from accumulated failure patterns.
    #[cfg(feature = "school_learning")]
    fn generate_lessons_from_failures(&self) -> Vec<crate::school::code_learning::CodeLesson> {
        let failures: Vec<(String, String, usize)> = self
            .failure_patterns
            .iter()
            .map(|(pattern, count)| (pattern.clone(), self.task.clone(), *count))
            .collect();
        crate::school::code_learning::lessons_from_failures(&failures, 5)
    }

    /// Parse structured test failures from cargo test stderr output.
    pub(super) fn parse_test_failures(stderr: &str) -> Vec<StructuredTestFailure> {
        let mut failures = Vec::new();
        let mut current_test: Option<String> = None;
        let mut current_output = String::new();

        for line in stderr.lines() {
            if line.starts_with("---- ") && line.ends_with(" stdout ----") {
                if let Some(ref name) = current_test {
                    failures.push(Self::build_test_failure(name, &current_output));
                }
                let name = line
                    .trim_start_matches("---- ")
                    .trim_end_matches(" stdout ----")
                    .to_string();
                current_test = Some(name);
                current_output.clear();
            } else if current_test.is_some() {
                current_output.push_str(line);
                current_output.push('\n');
            }
        }
        if let Some(ref name) = current_test {
            failures.push(Self::build_test_failure(name, &current_output));
        }
        failures
    }

    /// Build a structured test failure from a test name and its captured output.
    pub(super) fn build_test_failure(test_name: &str, output: &str) -> StructuredTestFailure {
        let (kind, expected, actual) = if output.contains("assertion `left == right` failed") {
            let left = output
                .lines()
                .find(|l| l.trim().starts_with("left:"))
                .map(|l| l.trim().trim_start_matches("left:").trim().to_string());
            let right = output
                .lines()
                .find(|l| l.trim().starts_with("right:"))
                .map(|l| l.trim().trim_start_matches("right:").trim().to_string());
            (TestFailureKind::AssertEq, right, left)
        } else if output.contains("assertion") && output.contains("failed") {
            (TestFailureKind::Assert, None, None)
        } else if output.contains("panicked at") {
            (TestFailureKind::Panic, None, None)
        } else {
            (TestFailureKind::Other, None, None)
        };

        let message = output
            .lines()
            .find(|l| l.contains("panicked at") || l.contains("assertion"))
            .map(|l| l.trim().to_string());

        let (file, line) = output
            .lines()
            .find(|l| l.contains(".rs:"))
            .map(|l| Self::extract_panic_location(l))
            .unwrap_or((None, None));

        StructuredTestFailure {
            test_name: test_name.to_string(),
            failure_kind: kind,
            expected,
            actual,
            message,
            file,
            line,
        }
    }

    /// Extract file path and line number from a panic location string.
    pub(super) fn extract_panic_location(location: &str) -> (Option<String>, Option<usize>) {
        let loc = location.trim().trim_start_matches("at ");
        if let Some(colon_idx) = loc.rfind(".rs:") {
            let file_end = colon_idx + 3;
            let file = loc[..file_end].trim().to_string();
            let after = &loc[file_end + 1..];
            let line = after.split(':').next().and_then(|s| s.parse().ok());
            (Some(file), line)
        } else {
            (None, None)
        }
    }

    /// Format structured test failures into a prompt-friendly string.
    pub(super) fn format_structured_test_failures(failures: &[StructuredTestFailure]) -> String {
        if failures.is_empty() {
            return String::new();
        }
        let mut out = format!("\n{} test failure(s):\n", failures.len());
        for f in failures {
            out.push_str(&format!("  - {} ({:?})", f.test_name, f.failure_kind));
            if let (Some(exp), Some(act)) = (&f.expected, &f.actual) {
                out.push_str(&format!(" expected={}, got={}", exp, act));
            }
            if let (Some(file), Some(line)) = (&f.file, f.line) {
                out.push_str(&format!(" at {}:{}", file, line));
            }
            if let Some(msg) = &f.message {
                let short: String = msg.chars().take(100).collect();
                out.push_str(&format!(" \u{2014} {}", short));
            }
            out.push('\n');
        }
        out
    }

    /// Extract consciousness signals from a cycle result for decision-making.
    pub(super) fn extract_consciousness_signals(
        &self,
        cycle_result: &CycleResult,
    ) -> ConsciousnessSignals {
        let prediction_error = 1.0 - self.cognitive_loop.prediction_confidence();
        let confidence_velocity = if self.prediction_error_history.len() >= 2 {
            let prev = self.prediction_error_history[self.prediction_error_history.len() - 1];
            self.cognitive_loop.prediction_confidence() - (1.0 - prev)
        } else {
            0.0
        };
        let phi = cycle_result.metadata.consciousness.consciousness_level as f32;
        let phi_slope = if self.phi_trace.len() >= 2 {
            let last = self.phi_trace[self.phi_trace.len() - 1];
            phi - last
        } else {
            0.0
        };
        let fep_surprise = cycle_result.metadata.fep.fep_surprise;

        ConsciousnessSignals {
            prediction_error,
            confidence_velocity,
            phi,
            phi_slope,
            fep_surprise,
        }
    }

    /// Backfill error_knowledge with test success after tests pass.
    ///
    /// When a fix is recorded via `store_fix_strategies()`, `tests_passed` is set to
    /// `None` because we don't know yet. Once tests actually pass, this method
    /// re-records all recent facts with `tests_passed: Some(true)`, improving the
    /// Bayesian success rate for those fix strategies.
    pub(super) fn backfill_error_knowledge_test_success(&mut self) {
        // Get the facts recorded during this run (recent ones with tests_passed=None)
        let facts_to_update: Vec<(String, super::error_knowledge::ErrorCategory, String)> = self
            .error_knowledge
            .recent_facts()
            .iter()
            .filter(|f| f.tests_passed.is_none() && f.compiled)
            .map(|f| (f.error_code.clone(), f.category, f.fix_strategy.clone()))
            .collect();

        for (error_code, category, fix_strategy) in facts_to_update {
            self.error_knowledge
                .record_fix(super::error_knowledge::CodeErrorFact {
                    error_code,
                    category,
                    pattern_signature: String::new(), // already indexed by code
                    fix_strategy,
                    compiled: true,
                    tests_passed: Some(true),
                    context_snippet: String::new(),
                });
        }

        if !self.error_knowledge.recent_facts().is_empty() {
            tracing::info!(
                target: "symthaea::coding_agent",
                total_facts = self.error_knowledge.total_facts(),
                distinct_codes = self.error_knowledge.distinct_codes(),
                "Backfilled error knowledge with test success"
            );
        }
    }
}
