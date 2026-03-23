// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public accessor methods and event handling for the coding agent.

use super::*;

impl CodingAgent {
    /// Attach an event channel for streaming agent progress.
    pub fn with_event_channel(mut self) -> (Self, std::sync::mpsc::Receiver<AgentEvent>) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.event_sink = Some(tx);
        (self, rx)
    }

    /// Attach an event sink after construction.
    pub fn subscribe_events(&mut self, tx: std::sync::mpsc::Sender<AgentEvent>) {
        self.event_sink = Some(tx);
    }

    /// Emit an event to the event channel (no-op if no sink).
    pub(super) fn emit_event(&self, event: AgentEvent) {
        if let Some(ref sink) = self.event_sink {
            let _ = sink.send(event);
        }
    }

    /// Emit a phase transition event.
    pub(super) fn emit_phase_transition(&self, from: &TaskPhase, to: &TaskPhase) {
        self.emit_event(AgentEvent::PhaseTransition {
            from: from.clone(),
            to: to.clone(),
            iteration: self.iteration,
        });
    }

    /// Helper: emit retry strategy (avoids borrow issues with emit_event).
    pub(super) fn emit_state_changed(&self, strategy: &RetryStrategy) {
        self.emit_event(AgentEvent::RetryStrategyChanged(strategy.clone()));
    }

    /// Select the next retry strategy, cycling through options.
    pub(super) fn next_retry_strategy(&mut self) -> RetryStrategy {
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

    // -- Public API --

    /// Get the current phase.
    pub fn phase(&self) -> &TaskPhase {
        &self.phase
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Access the underlying cognitive loop for direct inspection.
    pub fn cognitive_loop(&self) -> &CognitiveLoopService {
        &self.cognitive_loop
    }

    /// Access the underlying cognitive loop mutably.
    pub fn cognitive_loop_mut(&mut self) -> &mut CognitiveLoopService {
        &mut self.cognitive_loop
    }

    /// Set a custom intelligent dispatcher for LLM-routed code generation.
    pub fn with_dispatcher(mut self, dispatcher: IntelligentDispatcher) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Set codebase context from an external CodebaseMemory query.
    pub fn set_code_context(&mut self, context: Vec<String>) {
        self.code_context = context;
    }

    /// Index a project directory into a `CodebaseMemory` and inject relevant context.
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

            let filename = path.file_name().and_then(|f| f.to_str());
            let ext = path.extension().and_then(|e| e.to_str());
            let supported = matches!(ext, Some("rs") | Some("py") | Some("nix"));
            if !supported {
                continue;
            }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

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
    pub(super) fn reindex_file(&mut self, path: &std::path::Path, source: &str) {
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
    pub(super) fn build_source_context(
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
                    "// {} \u{2014} {:?} `{}` (similarity: {:.3})\n{}",
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
    pub(super) fn extract_entity_source(
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

    /// Get the last dispatch result.
    pub fn last_dispatch(&self) -> Option<&DispatchResult> {
        self.last_dispatch.as_ref()
    }

    /// Get the dispatcher's total energy consumption.
    pub fn total_energy(&self) -> f64 {
        self.dispatcher.as_ref().map_or(0.0, |d| d.total_energy())
    }

    /// Get accumulated failure patterns from this run.
    pub fn failure_patterns(&self) -> &[(String, usize)] {
        &self.failure_patterns
    }

    /// Whether the agent has a persistent experience store.
    pub fn has_experience_store(&self) -> bool {
        self.experience_store.is_some()
    }

    /// Get the count of stored experiences.
    pub fn experience_count(&self) -> usize {
        if let Some(ref store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(store.count());
            }
        }
        0
    }

    /// Get cached success patterns from the experience store.
    pub fn cached_successes(&self) -> Vec<(String, String)> {
        self.experience_store
            .as_ref()
            .map(|s| s.cached_successes().to_vec())
            .unwrap_or_default()
    }

    /// Get cached error hints from the experience store.
    pub fn cached_error_hints(&self) -> Vec<(String, String)> {
        self.experience_store
            .as_ref()
            .map(|s| s.cached_error_hints().to_vec())
            .unwrap_or_default()
    }

    /// Get the current execution plan profile (if any).
    pub fn current_plan_profile(&self) -> Option<&PlanProfile> {
        self.current_plan.as_ref()
    }

    /// Get remaining energy budget.
    pub fn remaining_energy(&self) -> f32 {
        self.energy_budget
    }

    /// Build a plan for a hypothetical action and evaluate it.
    pub fn evaluate_hypothetical_plan(&self, plan: &Molecule) -> (bool, String, PlanProfile) {
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        let profile = plan.profile();
        let (approved, reason) = self.evaluate_plan(plan, current_phi);
        (approved, reason, profile)
    }
}
