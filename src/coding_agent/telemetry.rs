use super::types::*;
use super::CodingAgent;
use crate::coding_experience::CodingExperience;
use crate::cognitive_loop::CycleResult;
use crate::language::intelligent_dispatcher::BackendTier;
use crate::mind::structured_thought::EpistemicStatus;
use std::path::PathBuf;

impl CodingAgent {
    /// Parse structured test failures from cargo test stderr output.
    pub(crate) fn parse_test_failures(stderr: &str) -> Vec<StructuredTestFailure> {
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
    pub(crate) fn build_test_failure(test_name: &str, output: &str) -> StructuredTestFailure {
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
    pub(crate) fn extract_panic_location(location: &str) -> (Option<String>, Option<usize>) {
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
    pub(crate) fn format_structured_test_failures(failures: &[StructuredTestFailure]) -> String {
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
                out.push_str(&format!(" — {}", short));
            }
            out.push('\n');
        }
        out
    }

    /// Extract consciousness signals from a cycle result for decision-making.
    pub(crate) fn extract_consciousness_signals(
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

    /// Emit an event to the event channel (no-op if no sink).
    pub(crate) fn emit_event(&self, event: AgentEvent) {
        if let Some(ref sink) = self.event_sink {
            let _ = sink.send(event);
        }
    }

    /// Emit a phase transition event.
    pub(crate) fn emit_phase_transition(&self, from: &TaskPhase, to: &TaskPhase) {
        self.emit_event(AgentEvent::PhaseTransition {
            from: from.clone(),
            to: to.clone(),
            iteration: self.iteration,
        });
    }

    /// Helper: emit retry strategy (avoids borrow issues with emit_event).
    pub(crate) fn emit_state_changed(&self, strategy: &RetryStrategy) {
        self.emit_event(AgentEvent::RetryStrategyChanged(strategy.clone()));
    }

    /// Map prediction confidence to epistemic status.
    pub(crate) fn confidence_to_epistemic(confidence: f32) -> EpistemicStatus {
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

    /// Normalize an error message for pattern matching (strip paths, line numbers).
    pub(crate) fn normalize_error_pattern(error: &str) -> String {
        let mut normalized = String::new();
        for line in error.lines().take(3) {
            if line.contains("error[E") || line.contains("error:") {
                normalized.push_str(line.trim());
                normalized.push(' ');
            }
        }
        if normalized.is_empty() {
            error.lines().next().unwrap_or(error).to_string()
        } else {
            normalized.trim().to_string()
        }
    }

    /// Store a coding experience (success or failure) in the persistent store.
    pub(crate) fn store_experience(&mut self, detail: &str, success: bool) {
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
    pub(crate) fn record_generation_outcome(&mut self, success: bool) {
        let tier = self.generation_tiers.last().copied();
        if let (Some(tier), Some(ref mut dispatcher)) = (tier, &mut self.dispatcher) {
            dispatcher.record_outcome(tier, success);
        }

        if success {
            if let Some(code) = self.generated_code.clone() {
                let summary: String = code.chars().take(200).collect();
                self.store_experience(&summary, true);

                if let Some((last_error, _)) = self.failure_patterns.last().cloned() {
                    self.store_fix_hint(&last_error, &code);
                }

                if let Some(ref mut store) = self.experience_store {
                    store.store_learned_template(&self.task, &code);
                    tracing::info!(
                        target: "symthaea::coding_agent",
                        task = %self.task,
                        code_len = code.len(),
                        tier = ?tier,
                        "Distilled generation into learned template"
                    );
                }
            }
        }
    }

    /// Store a fix hint.
    pub(crate) fn store_fix_hint(&mut self, error_pattern: &str, fix_code: &str) {
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
    pub(crate) fn store_execution_trace(&mut self, trace: &[(String, f32, String)]) {
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
        let recipe_key = atom_names.join("→");

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

    /// Select the next retry strategy, cycling through options.
    pub(crate) fn next_retry_strategy(&mut self) -> RetryStrategy {
        let strategies = [
            RetryStrategy::DifferentTemplate,
            RetryStrategy::DifferentBackend(BackendTier::LocalLlm),
            RetryStrategy::DifferentBackend(BackendTier::CloudLlm),
            RetryStrategy::SimplifyScope,
            RetryStrategy::RequestClarification(
                "Unable to resolve after multiple strategies. Could you clarify or simplify the task?".to_string(),
            ),
        ];

        for s in &strategies {
            if !self.retry_state.strategies_tried.contains(s) {
                let strategy = s.clone();
                self.retry_state.strategies_tried.push(strategy.clone());
                self.emit_event(AgentEvent::RetryStrategyChanged(strategy.clone()));
                return strategy;
            }
        }

        let fallback =
            RetryStrategy::RequestClarification("All retry strategies exhausted.".to_string());
        self.emit_state_changed(&fallback);
        fallback
    }

    /// Build HDC context prompt from indexed codebase memory.
    #[cfg(feature = "code_generation")]
    pub(crate) fn build_hdc_context_prompt(&self) -> String {
        use crate::hdc::code_encoder::CodeHDEncoder;

        let code_memory = match &self.code_memory {
            Some(m) => m,
            None => return String::new(),
        };

        let encoder = CodeHDEncoder::new(16_384);
        let query_hv = encoder.encode_name(&self.task);
        let matches = code_memory.query(&query_hv, 5);

        if matches.is_empty() {
            return String::new();
        }

        let coherence = code_memory.codebase_coherence();
        let mut prompt = format!(
            "## Codebase context (HDC similarity search, coherence={:.2})\n",
            coherence
        );
        for m in &matches {
            prompt.push_str(&format!("- {} (similarity={:.3})\n", m.name, m.similarity));
            if let Some(src) = self.source_cache.get(&m.path) {
                let snippet = Self::extract_entity_source(src, &m.name, m.kind);
                if !snippet.contains("(source not found)") {
                    let truncated: String = snippet.chars().take(200).collect();
                    prompt.push_str(&format!("  ```\n  {}\n  ```\n", truncated));
                }
            }
        }
        prompt
    }

    #[cfg(not(feature = "code_generation"))]
    pub(crate) fn build_hdc_context_prompt(&self) -> String {
        String::new()
    }

    /// Dynamically query CodebaseMemory for context relevant to current errors.
    #[cfg(feature = "code_generation")]
    pub(crate) fn build_dynamic_error_context(&self) -> String {
        let code_memory = match (&self.phase, &self.code_memory) {
            (TaskPhase::Fixing, Some(m)) => m,
            _ => return String::new(),
        };
        let test_output = match &self.last_test_output {
            Some(o) => o,
            None => return String::new(),
        };

        let mut query_terms: Vec<String> = Vec::new();
        for line in test_output.lines() {
            let mut rest = line;
            while let Some(start) = rest.find('`') {
                let after = &rest[start + 1..];
                if let Some(end) = after.find('`') {
                    let name = &after[..end];
                    let clean = name.trim_end_matches("()");
                    if !clean.is_empty()
                        && clean.len() < 60
                        && !clean.contains(' ')
                        && !clean.starts_with("error")
                        && !clean.starts_with("help")
                    {
                        if !query_terms.iter().any(|t| t == clean) {
                            query_terms.push(clean.to_string());
                        }
                    }
                    rest = &after[end + 1..];
                } else {
                    break;
                }
            }
        }

        if query_terms.is_empty() {
            return String::new();
        }

        let encoder = code_memory.encoder();
        let mut seen_names = std::collections::HashSet::new();
        let mut result = String::new();

        for term in query_terms.iter().take(5) {
            let query_hv = encoder.encode_name(term);
            let matches = code_memory.query(&query_hv, 3);

            for m in &matches {
                if m.similarity < 0.25 || !seen_names.insert(m.name.clone()) {
                    continue;
                }
                if let Some(src) = self.source_cache.get(&m.path) {
                    let snippet = Self::extract_entity_source(src, &m.name, m.kind);
                    if !snippet.contains("(source not found)") {
                        let truncated: String = snippet.chars().take(300).collect();
                        result.push_str(&format!(
                            "- `{}` ({:?}, {}): ```\n{}\n```\n",
                            m.name,
                            m.kind,
                            m.path.display(),
                            truncated
                        ));
                    }
                } else {
                    result.push_str(&format!(
                        "- `{}` ({:?}, {}, sim={:.2})\n",
                        m.name,
                        m.kind,
                        m.path.display(),
                        m.similarity
                    ));
                }
            }
        }

        result
    }

    #[cfg(not(feature = "code_generation"))]
    pub(crate) fn build_dynamic_error_context(&self) -> String {
        String::new()
    }

    /// Build the final result.
    pub(crate) fn build_result(&self) -> AgentResult {
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
    pub(crate) fn generate_lessons_from_failures(
        &self,
    ) -> Vec<crate::school::code_learning::CodeLesson> {
        let failures: Vec<(String, String, usize)> = self
            .failure_patterns
            .iter()
            .map(|(pattern, count)| (pattern.clone(), self.task.clone(), *count))
            .collect();
        crate::school::code_learning::lessons_from_failures(&failures, 5)
    }

    /// Index a project directory into a `CodebaseMemory`.
    #[cfg(feature = "code_generation")]
    pub fn index_project(
        &mut self,
        root: &std::path::Path,
    ) -> anyhow::Result<(usize, usize, usize)> {
        use crate::hdc::code_encoder::CodeHDEncoder;
        use crate::hdc::code_memory::CodebaseMemory;
        use crate::language::parser_registry::ParserRegistry;
        use ignore::WalkBuilder;

        let mut memory = CodebaseMemory::with_default_encoder();
        let mut parser_registry = ParserRegistry::with_builtins();
        let mut files_indexed = 0usize;
        let mut parse_errors = 0usize;

        for entry in WalkBuilder::new(root)
            .hidden(true)
            .git_ignore(true)
            .build()
            .flatten()
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let ext = path.extension().and_then(|e| e.to_str());
            let supported = matches!(ext, Some("rs") | Some("py") | Some("nix"));
            if !supported {
                continue;
            }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let filename = path.file_name().and_then(|f| f.to_str());
            match parser_registry.parse(&source, None, filename) {
                Ok(parsed) => {
                    memory.index_file(path, &parsed);
                    files_indexed += 1;
                    self.source_cache.insert(path.to_path_buf(), source);
                }
                Err(_) => {
                    parse_errors += 1;
                }
            }
        }

        let stats = memory.stats();
        tracing::info!(
            target: "symthaea::coding_agent",
            files = files_indexed,
            functions = stats.functions,
            types = stats.types,
            parse_errors = parse_errors,
            "Indexed project into CodebaseMemory"
        );

        if !self.task.is_empty() {
            self.code_context = Self::build_source_context(&memory, &self.source_cache, &self.task);
        }

        self.code_memory = Some(memory);

        Ok((files_indexed, stats.functions, stats.types))
    }

    /// Re-index a single file after it has been written/modified.
    #[cfg(feature = "code_generation")]
    pub(crate) fn reindex_file(&mut self, path: &std::path::Path, source: &str) {
        use crate::language::parser_registry::ParserRegistry;
        let filename = path.file_name().and_then(|f| f.to_str());
        let mut parser_registry = ParserRegistry::with_builtins();
        if let Ok(parsed) = parser_registry.parse(source, None, filename) {
            if let Some(ref mut memory) = self.code_memory {
                memory.update_file(path, &parsed);
            }
        }
        self.source_cache
            .insert(path.to_path_buf(), source.to_string());
    }

    /// Build source-level context from CodebaseMemory matches.
    #[cfg(feature = "code_generation")]
    pub(crate) fn build_source_context(
        memory: &crate::hdc::code_memory::CodebaseMemory,
        source_cache: &std::collections::HashMap<PathBuf, String>,
        task: &str,
    ) -> Vec<String> {
        let encoder = memory.encoder();
        let intent_hv = encoder.encode_name(task);
        let matches = memory.query(&intent_hv, 5);
        matches
            .iter()
            .filter(|m| m.similarity > 0.2)
            .filter_map(|m| {
                let source = source_cache.get(&m.path)?;
                let snippet = Self::extract_entity_source(source, &m.name, m.kind);
                Some(format!(
                    "// {} — {:?} `{}` (similarity: {:.3})\n{}",
                    m.path.display(),
                    m.kind,
                    m.name,
                    m.similarity,
                    snippet
                ))
            })
            .collect()
    }

    /// Extract source code for a named entity using brace-matching (up to 20 lines).
    #[cfg(feature = "code_generation")]
    pub(crate) fn extract_entity_source(
        source: &str,
        name: &str,
        kind: crate::language::code_parser::EntityKind,
    ) -> String {
        use crate::language::code_parser::EntityKind;
        let keyword = match kind {
            EntityKind::Function | EntityKind::Method => "fn ",
            EntityKind::Struct => "struct ",
            EntityKind::Enum => "enum ",
            EntityKind::Trait | EntityKind::Interface => "trait ",
            EntityKind::Class => "class ",
            _ => "fn ",
        };
        let pattern = format!("{keyword}{name}");
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.contains(&pattern) {
                let mut depth = 0i32;
                let mut out = Vec::new();
                let mut started = false;
                for j in i..lines.len().min(i + 30) {
                    out.push(lines[j]);
                    for ch in lines[j].chars() {
                        if ch == '{' {
                            depth += 1;
                            started = true;
                        }
                        if ch == '}' {
                            depth -= 1;
                        }
                    }
                    if started && depth <= 0 {
                        break;
                    }
                    if out.len() >= 20 {
                        out.push("    // ... (truncated)");
                        break;
                    }
                }
                return out.join("\n");
            }
        }
        format!("// {keyword}{name} (source not found)")
    }
}
